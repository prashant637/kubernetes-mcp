# server.py

import os
import json
import logging
import subprocess
import threading
import uuid
from datetime import datetime
from typing import Optional, Literal, List, Dict

from fastmcp import FastMCP
from pydantic import BaseModel, Field
from kubernetes import client, config
from kubernetes.client import ApiException

# --- Logging to STDERR (required for STDIO MCP servers) ---
logging.basicConfig(level=logging.INFO)  # goes to stderr by default

# --- Kubernetes client bootstrap ---
def load_kube():
    kubeconfig = os.getenv("KUBECONFIG")  # allow override
    context = os.getenv("KUBE_CONTEXT")
    if kubeconfig or context:
        # Out-of-cluster kubeconfig with optional context
        config.load_kube_config(config_file=kubeconfig, context=context)
    else:
        # Try default flow; if running in-cluster later, use:
        # config.load_incluster_config()
        config.load_kube_config()
    return client.CoreV1Api(), client.AppsV1Api()

core_v1, apps_v1 = load_kube()
DEFAULT_NS = os.getenv("KUBE_NAMESPACE", "default")

# --- CLI helpers & state ---
PORT_FORWARD_PROCS: Dict[str, subprocess.Popen] = {}
PORT_FORWARD_LOCK = threading.Lock()

def _format_command_result(cmd: List[str], proc: subprocess.CompletedProcess) -> Dict[str, object]:
    return {
        "command": " ".join(cmd),
        "return_code": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }

def run_cli(cmd: List[str], input_data: Optional[str] = None) -> Dict[str, object]:
    logging.info("Running command: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            input=input_data,
            capture_output=True,
            text=True,
            check=False,
        )
        return _format_command_result(cmd, proc)
    except FileNotFoundError:
        return {"error": f"Executable '{cmd[0]}' not found on PATH.", "command": " ".join(cmd)}

# --- MCP server ---
mcp = FastMCP("k8s-remote-mcp-server", version="1.0.0")


# ---------------------------
# Input / Output Schemas
# ---------------------------
class DeploymentSpec(BaseModel):
    name: str = Field(..., description="Deployment name")
    image: str = Field(..., description="Container image, e.g. nginx:1.25")
    replicas: int = Field(1, ge=0, description="Replica count")
    container_port: Optional[int] = Field(80, description="Container port")
    namespace: str = Field(DEFAULT_NS, description="Kubernetes namespace")
    labels: Dict[str, str] = Field(default_factory=lambda: {"app": "app"})


class ServiceSpec(BaseModel):
    name: str
    namespace: str = Field(DEFAULT_NS)
    type: Literal["ClusterIP", "NodePort", "LoadBalancer"] = "ClusterIP"
    port: int = 80
    target_port: int = 80
    selector: Dict[str, str] = Field(default_factory=lambda: {"app": "app"})


class PodSpec(BaseModel):
    name: str
    image: str
    namespace: str = Field(DEFAULT_NS)
    container_port: Optional[int] = 80
    labels: Dict[str, str] = Field(default_factory=lambda: {"app": "app"})


class NameNs(BaseModel):
    name: str
    namespace: str = Field(DEFAULT_NS)


class DiagnoseFixInput(BaseModel):
    pod_name: str
    namespace: str = Field(DEFAULT_NS)
    mode: Literal["plan", "apply"] = "plan"
    image_pull_secret: Optional[str] = None


class ListNamespace(BaseModel):
    namespace: str = Field(DEFAULT_NS, description="Kubernetes namespace")


class KubectlResourceInput(BaseModel):
    resource: str = Field(..., description="Resource type (e.g., pods, deployments, svc/app) or full resource reference (kind/name).")
    name: Optional[str] = Field(None, description="Object name (omit if resource already contains it).")
    namespace: Optional[str] = Field(None, description="Namespace to target; defaults to server default.")
    all_namespaces: bool = Field(False, description="If true, query across all namespaces.")
    output: Optional[str] = Field(None, description="kubectl -o output format, e.g., json, yaml, wide.")
    additional_args: List[str] = Field(default_factory=list, description="Extra kubectl flags.")


class KubectlManifestInput(BaseModel):
    manifest: str = Field(..., description="YAML manifest content.")
    namespace: Optional[str] = Field(None, description="Optional namespace override.")
    server_side: bool = Field(False, description="Use server-side apply.")
    force_conflicts: bool = Field(False, description="Force conflicts when using server-side apply.")
    field_manager: Optional[str] = Field(None, description="Field manager name.")
    validate: bool = Field(True, description="Validate the manifest before applying.")
    dry_run: Optional[Literal["client", "server"]] = Field(None, description="Dry-run mode.")
    additional_args: List[str] = Field(default_factory=list)


class KubectlPatchInput(BaseModel):
    resource: str
    name: str
    namespace: str = Field(DEFAULT_NS)
    patch: str = Field(..., description="Patch payload (JSON/YAML).")
    patch_type: Literal["json", "merge", "strategic", "apply"] = "merge"
    additional_args: List[str] = Field(default_factory=list)


class KubectlRolloutInput(BaseModel):
    action: Literal["status", "history", "restart", "undo", "pause", "resume"]
    resource: str = Field(..., description="Workload reference, e.g., deployment/my-app")
    namespace: str = Field(DEFAULT_NS)
    revision: Optional[int] = None
    additional_args: List[str] = Field(default_factory=list)


class KubectlScaleInput(BaseModel):
    resource: str = Field(..., description="Workload reference, e.g., deployment/my-app")
    replicas: int = Field(..., ge=0)
    namespace: Optional[str] = Field(None)
    additional_args: List[str] = Field(default_factory=list)


class KubectlLogsInput(BaseModel):
    pod: str
    namespace: str = Field(DEFAULT_NS)
    container: Optional[str] = None
    tail_lines: Optional[int] = Field(None, ge=1)
    since_seconds: Optional[int] = Field(None, ge=1)
    previous: bool = Field(False)
    timestamps: bool = Field(False)
    additional_args: List[str] = Field(default_factory=list)


class KubectlExecInput(BaseModel):
    pod: str
    namespace: str = Field(DEFAULT_NS)
    container: Optional[str] = None
    command: List[str] = Field(default_factory=list, description="Command to execute inside the container.")
    tty: bool = Field(False)
    stdin: bool = Field(False)
    additional_args: List[str] = Field(default_factory=list)


class PortForwardInput(BaseModel):
    resource: Literal["pod", "svc", "deployment", "statefulset", "daemonset"] = "pod"
    name: str = Field(..., description="Resource name to port-forward.")
    namespace: str = Field(DEFAULT_NS)
    ports: List[str] = Field(..., description="Port mappings, e.g., ['8080:80'].")
    address: Optional[str] = Field(None, description="Listening address, e.g., 0.0.0.0")
    additional_args: List[str] = Field(default_factory=list)


class PortForwardSessionInput(BaseModel):
    session_id: str = Field(..., description="Session ID returned when port-forward started.")


class TopPodsInput(BaseModel):
    namespace: Optional[str] = Field(None, description="Namespace scope; omit for current default.")
    all_namespaces: bool = Field(False)
    no_headers: bool = Field(True)
    sort_by: Optional[str] = Field(None, description="Field to sort by, e.g., cpu")


class ContextInput(BaseModel):
    action: Literal["current", "list", "use", "set-namespace"]
    context: Optional[str] = Field(None, description="Target context name for 'use'.")
    namespace: Optional[str] = Field(None, description="Namespace to set for 'set-namespace'.")
    additional_args: List[str] = Field(default_factory=list)


class EventsInput(BaseModel):
    namespace: Optional[str] = Field(DEFAULT_NS, description="Namespace to query; omit for default.")
    all_namespaces: bool = Field(False)
    field_selector: Optional[str] = None
    limit: Optional[int] = Field(None, ge=1)


class HelmInstallInput(BaseModel):
    release_name: str = Field(..., description="Release name (e.g., myapp).")
    chart: str = Field(..., description="Chart reference (e.g., bitnami/nginx).")
    namespace: str = Field(DEFAULT_NS)
    create_namespace: bool = Field(False)
    version: Optional[str] = Field(None, description="Chart version.")
    values_yaml: Optional[str] = Field(None, description="Inline values.yaml content.")
    set: Dict[str, str] = Field(default_factory=dict, description="Key/value overrides for --set.")
    additional_args: List[str] = Field(default_factory=list)


class HelmReleaseQuery(BaseModel):
    namespace: Optional[str] = Field(None)
    all_namespaces: bool = Field(False)
    filter: Optional[str] = Field(None)
    output: Optional[str] = Field(None)
    additional_args: List[str] = Field(default_factory=list)


class HelmUninstallInput(BaseModel):
    release_name: str
    namespace: Optional[str] = Field(None)
    keep_history: bool = Field(False)
    additional_args: List[str] = Field(default_factory=list)


class HelmRepoInput(BaseModel):
    action: Literal["add", "update", "remove", "list"]
    name: Optional[str] = Field(None, description="Repository name (required for add/remove).")
    url: Optional[str] = Field(None, description="Repository URL (required for add).")
    additional_args: List[str] = Field(default_factory=list)


# ---------------------------
# Helper builders
# ---------------------------
def build_deployment(spec: DeploymentSpec):
    container = client.V1Container(
        name=spec.name,
        image=spec.image,
        ports=[client.V1ContainerPort(container_port=spec.container_port)] if spec.container_port else None,
    )
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels=spec.labels),
        spec=client.V1PodSpec(containers=[container]),
    )
    deployment_spec = client.V1DeploymentSpec(
        replicas=spec.replicas,
        selector=client.V1LabelSelector(match_labels=spec.labels),
        template=template,
    )
    return client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=client.V1ObjectMeta(name=spec.name, labels=spec.labels),
        spec=deployment_spec,
    )


def build_service(spec: ServiceSpec):
    svc_spec = client.V1ServiceSpec(
        type=spec.type,
        selector=spec.selector,
        ports=[client.V1ServicePort(port=spec.port, target_port=spec.target_port)],
    )
    return client.V1Service(
        api_version="v1",
        kind="Service",
        metadata=client.V1ObjectMeta(name=spec.name),
        spec=svc_spec,
    )


def build_pod(spec: PodSpec):
    container = client.V1Container(
        name=spec.name,
        image=spec.image,
        ports=[client.V1ContainerPort(container_port=spec.container_port)] if spec.container_port else None,
    )
    pod_spec = client.V1PodSpec(containers=[container])
    return client.V1Pod(
        api_version="v1",
        kind="Pod",
        metadata=client.V1ObjectMeta(name=spec.name, labels=spec.labels),
        spec=pod_spec,
    )


# ---------------------------
# Tools: Deployments
# ---------------------------
@mcp.tool(description="Create a Deployment")
def create_deployment(spec: DeploymentSpec) -> str:
    dep = build_deployment(spec)
    try:
        apps_v1.create_namespaced_deployment(namespace=spec.namespace, body=dep)
        return f"Deployment '{spec.name}' created in ns '{spec.namespace}'."
    except ApiException as e:
        return f"ERROR creating deployment: {e}"


@mcp.tool(description="Delete a Deployment")
def delete_deployment(inp: NameNs) -> str:
    try:
        apps_v1.delete_namespaced_deployment(name=inp.name, namespace=inp.namespace)
        return f"Deployment '{inp.name}' deleted from ns '{inp.namespace}'."
    except ApiException as e:
        return f"ERROR deleting deployment: {e}"


# ---------------------------
# Tools: Services
# ---------------------------
@mcp.tool(description="Create a Service")
def create_service(spec: ServiceSpec) -> str:
    svc = build_service(spec)
    try:
        core_v1.create_namespaced_service(namespace=spec.namespace, body=svc)
        return f"Service '{spec.name}' ({spec.type}) created in ns '{spec.namespace}'."
    except ApiException as e:
        return f"ERROR creating service: {e}"


@mcp.tool(description="Delete a Service")
def delete_service(inp: NameNs) -> str:
    try:
        core_v1.delete_namespaced_service(name=inp.name, namespace=inp.namespace)
        return f"Service '{inp.name}' deleted from ns '{inp.namespace}'."
    except ApiException as e:
        return f"ERROR deleting service: {e}"


# ---------------------------
# Tools: Pods (one-off pods)
# ---------------------------
@mcp.tool(description="Create a single Pod (for quick tests)")
def create_pod(spec: PodSpec) -> str:
    pod = build_pod(spec)
    try:
        core_v1.create_namespaced_pod(namespace=spec.namespace, body=pod)
        return f"Pod '{spec.name}' created in ns '{spec.namespace}'."
    except ApiException as e:
        return f"ERROR creating pod: {e}"


@mcp.tool(description="Delete a Pod")
def delete_pod(inp: NameNs) -> str:
    try:
        core_v1.delete_namespaced_pod(name=inp.name, namespace=inp.namespace)
        return f"Pod '{inp.name}' deleted from ns '{inp.namespace}'."
    except ApiException as e:
        return f"ERROR deleting pod: {e}"


# ---------------------------
# Tool: Diagnose & (optionally) Fix pod issues
# ---------------------------
@mcp.tool(description="Diagnose a pod and plan/apply common fixes for CrashLoopBackOff, ImagePullBackOff, etc.")
def diagnose_and_fix_pod(inp: DiagnoseFixInput) -> dict:
    name, ns = inp.pod_name, inp.namespace
    result: Dict[str, object] = {"pod": f"{ns}/{name}", "mode": inp.mode, "findings": [], "actions": []}

    try:
        pod = core_v1.read_namespaced_pod(name=name, namespace=ns)
        pod_status = pod.status
    except ApiException as e:
        return {"error": f"Unable to read pod: {e}"}

    # Collect recent events for the pod
    try:
        field_selector = f"involvedObject.kind=Pod,involvedObject.name={name}"
        events = core_v1.list_namespaced_event(ns, field_selector=field_selector).items
        result["events"] = [f"{ev.type} {ev.reason}: {ev.message}" for ev in events[-10:]]
    except Exception as e:
        result["events_error"] = f"Unable to fetch events: {e}"

    # Add logs from first container to help diagnosis
    try:
        containers = pod.spec.containers or []
        if containers:
            log = core_v1.read_namespaced_pod_log(name=name, namespace=ns, container=containers[0].name, tail_lines=200)
            result["logs_tail"] = log.splitlines()[-50:]
    except Exception as e:
        result["logs_error"] = f"Unable to fetch logs: {e}"

    # Heuristics for common issues
    reasons = []
    for cs in (pod_status.container_statuses or []):
        state = cs.state
        if state.waiting and state.waiting.reason:
            reasons.append(state.waiting.reason)
        if state.terminated and state.terminated.reason:
            reasons.append(state.terminated.reason)
    result["reasons"] = reasons

    planned: List[str] = []

    if any(r in ("ImagePullBackOff", "ErrImagePull") for r in reasons):
        planned.append("Verify image name/tag & registry access; if private, ensure imagePullSecret is configured.")
        if inp.image_pull_secret:
            planned.append(f"Attach imagePullSecret '{inp.image_pull_secret}' to the pod template (deployment) and restart.")
    if "CrashLoopBackOff" in reasons:
        planned.append("Inspect container logs for crash root cause; consider rollout restart or deleting the pod to reschedule.")
    if not reasons and pod_status.phase == "Pending":
        planned.append("Check scheduling constraints, node resources, PVC binding, taints/tolerations.")

    result["findings"] = planned

    if inp.mode == "apply":
        owner = (pod.metadata.owner_references or [None])[0]
        try:
            if owner and owner.kind in ("ReplicaSet",):
                core_v1.delete_namespaced_pod(name=name, namespace=ns)
                result["actions"].append("Deleted pod to trigger reschedule from ReplicaSet.")
            if owner and owner.kind == "ReplicaSet":
                rs = apps_v1.read_namespaced_replica_set(owner.name, ns)
                rs_owner = (rs.metadata.owner_references or [None])[0]
                if rs_owner and rs_owner.kind == "Deployment":
                    ts = datetime.utcnow().isoformat() + "Z"
                    patch = {
                        "spec": {
                            "template": {
                                "metadata": {
                                    "annotations": {
                                        "kubectl.kubernetes.io/restartedAt": ts
                                    }
                                }
                            }
                        }
                    }
                    apps_v1.patch_namespaced_deployment(name=rs_owner.name, namespace=ns, body=patch)
                    result["actions"].append(f"Patched Deployment '{rs_owner.name}' to rollout restart.")
            if inp.image_pull_secret and owner and owner.kind == "ReplicaSet":
                rs = apps_v1.read_namespaced_replica_set(owner.name, ns)
                rs_owner = (rs.metadata.owner_references or [None])[0]
                if rs_owner and rs_owner.kind == "Deployment":
                    dep = apps_v1.read_namespaced_deployment(rs_owner.name, ns)
                    spec = dep.spec.template.spec
                    ips = spec.image_pull_secrets or []
                    names = {x.name for x in ips}
                    if inp.image_pull_secret not in names:
                        ips.append(client.V1LocalObjectReference(name=inp.image_pull_secret))
                        spec.image_pull_secrets = ips
                        apps_v1.patch_namespaced_deployment(rs_owner.name, ns, dep)
                        result["actions"].append(f"Added imagePullSecret '{inp.image_pull_secret}' to Deployment '{rs_owner.name}'.")
        except ApiException as e:
            result["actions"].append(f"ERROR applying fix: {e}")

    return result


# ---------------------------
# Tools: List Pods / Deployments / Services
# ---------------------------
@mcp.tool(description="List all Pods in a namespace")
def list_pods(inp: ListNamespace) -> list[dict]:
    try:
        pods = core_v1.list_namespaced_pod(namespace=inp.namespace).items
        return [
            {
                "name": p.metadata.name,
                "phase": p.status.phase,
                "node": p.spec.node_name,
                "restarts": sum(cs.restart_count for cs in (p.status.container_statuses or [])),
                "age": p.metadata.creation_timestamp.isoformat()
            }
            for p in pods
        ]
    except ApiException as e:
        return [{"error": f"ERROR listing pods: {e}"}]


@mcp.tool(description="List all Deployments in a namespace")
def list_deployments(inp: ListNamespace) -> list[dict]:
    try:
        deps = apps_v1.list_namespaced_deployment(namespace=inp.namespace).items
        return [
            {
                "name": d.metadata.name,
                "replicas": d.status.replicas or 0,
                "available": d.status.available_replicas or 0,
                "age": d.metadata.creation_timestamp.isoformat()
            }
            for d in deps
        ]
    except ApiException as e:
        return [{"error": f"ERROR listing deployments: {e}"}]


@mcp.tool(description="List all Services in a namespace")
def list_services(inp: ListNamespace) -> list[dict]:
    try:
        svcs = core_v1.list_namespaced_service(namespace=inp.namespace).items
        return [
            {
                "name": s.metadata.name,
                "type": s.spec.type,
                "cluster_ip": s.spec.cluster_ip,
                "ports": [{"port": p.port, "targetPort": p.target_port} for p in s.spec.ports],
                "age": s.metadata.creation_timestamp.isoformat()
            }
            for s in svcs
        ]
    except ApiException as e:
        return [{"error": f"ERROR listing services: {e}"}]


# ---------------------------
# Additional kubectl utilities
# ---------------------------
@mcp.tool(description="Run 'kubectl get' for arbitrary resources.")
def kubectl_get(inp: KubectlResourceInput) -> Dict[str, object]:
    cmd = ["kubectl", "get", inp.resource]
    if inp.name:
        cmd.append(inp.name)
    if inp.all_namespaces:
        cmd.append("-A")
    elif inp.namespace:
        cmd += ["-n", inp.namespace]
    if inp.output:
        cmd += ["-o", inp.output]
    cmd += inp.additional_args
    return run_cli(cmd)


@mcp.tool(description="Run 'kubectl describe' on a resource.")
def kubectl_describe(inp: KubectlResourceInput) -> Dict[str, object]:
    cmd = ["kubectl", "describe", inp.resource]
    if inp.name:
        cmd.append(inp.name)
    if inp.all_namespaces:
        cmd.append("-A")
    elif inp.namespace:
        cmd += ["-n", inp.namespace]
    cmd += inp.additional_args
    return run_cli(cmd)


@mcp.tool(description="Apply Kubernetes manifest(s) via kubectl apply.")
def kubectl_apply(inp: KubectlManifestInput) -> Dict[str, object]:
    cmd = ["kubectl", "apply", "-f", "-"]
    if inp.namespace:
        cmd += ["-n", inp.namespace]
    if inp.server_side:
        cmd.append("--server-side")
    if inp.force_conflicts:
        cmd.append("--force-conflicts")
    if inp.field_manager:
        cmd += ["--field-manager", inp.field_manager]
    if not inp.validate:
        cmd.append("--validate=false")
    if inp.dry_run:
        cmd += ["--dry-run", inp.dry_run]
    cmd += inp.additional_args
    return run_cli(cmd, input_data=inp.manifest)


@mcp.tool(description="Create Kubernetes resources from manifest(s) via kubectl create.")
def kubectl_create(inp: KubectlManifestInput) -> Dict[str, object]:
    cmd = ["kubectl", "create", "-f", "-"]
    if inp.namespace:
        cmd += ["-n", inp.namespace]
    if inp.dry_run:
        cmd += ["--dry-run", inp.dry_run]
    cmd += inp.additional_args
    return run_cli(cmd, input_data=inp.manifest)


@mcp.tool(description="Delete resources via kubectl delete.")
def kubectl_delete(inp: KubectlResourceInput) -> Dict[str, object]:
    cmd = ["kubectl", "delete", inp.resource]
    if inp.name:
        cmd.append(inp.name)
    if inp.all_namespaces:
        cmd.append("-A")
    elif inp.namespace:
        cmd += ["-n", inp.namespace]
    cmd += inp.additional_args
    return run_cli(cmd)


@mcp.tool(description="Patch a resource via kubectl patch.")
def kubectl_patch(inp: KubectlPatchInput) -> Dict[str, object]:
    cmd = ["kubectl", "patch", inp.resource, inp.name, "--type", inp.patch_type, "-p", inp.patch]
    if inp.namespace:
        cmd += ["-n", inp.namespace]
    cmd += inp.additional_args
    return run_cli(cmd)


@mcp.tool(description="Interact with kubectl rollout for workloads.")
def kubectl_rollout(inp: KubectlRolloutInput) -> Dict[str, object]:
    cmd = ["kubectl", "rollout", inp.action, inp.resource]
    if inp.action in {"status", "history"} and inp.revision:
        cmd += ["--revision", str(inp.revision)]
    if inp.namespace:
        cmd += ["-n", inp.namespace]
    cmd += inp.additional_args
    return run_cli(cmd)


@mcp.tool(description="Scale a workload via kubectl scale.")
def kubectl_scale(inp: KubectlScaleInput) -> Dict[str, object]:
    cmd = ["kubectl", "scale", inp.resource, f"--replicas={inp.replicas}"]
    if inp.namespace:
        cmd += ["-n", inp.namespace]
    cmd += inp.additional_args
    return run_cli(cmd)


@mcp.tool(description="List API resources (kubectl api-resources).")
def list_api_resources(additional_args: Optional[List[str]] = None) -> Dict[str, object]:
    cmd = ["kubectl", "api-resources"]
    if additional_args:
        cmd += additional_args
    return run_cli(cmd)


@mcp.tool(description="Retrieve a single Pod definition.")
def get_pod(inp: NameNs) -> Dict[str, object]:
    try:
        pod = core_v1.read_namespaced_pod(name=inp.name, namespace=inp.namespace)
        return client.ApiClient().sanitize_for_serialization(pod)
    except ApiException as e:
        return {"error": f"ERROR reading pod: {e}"}


@mcp.tool(description="Fetch pod/container logs via kubectl logs.")
def kubectl_logs(inp: KubectlLogsInput) -> Dict[str, object]:
    cmd = ["kubectl", "logs", inp.pod]
    if inp.container:
        cmd += ["-c", inp.container]
    if inp.namespace:
        cmd += ["-n", inp.namespace]
    if inp.tail_lines:
        cmd += ["--tail", str(inp.tail_lines)]
    if inp.since_seconds:
        cmd += ["--since", f"{inp.since_seconds}s"]
    if inp.previous:
        cmd.append("--previous")
    if inp.timestamps:
        cmd.append("--timestamps")
    cmd += inp.additional_args
    return run_cli(cmd)


@mcp.tool(description="Execute a command inside a pod (kubectl exec).")
def exec_in_pod(inp: KubectlExecInput) -> Dict[str, object]:
    if not inp.command:
        return {"error": "Command list cannot be empty."}
    cmd = ["kubectl", "exec"]
    if inp.tty:
        cmd.append("-it")
    if inp.stdin:
        cmd.append("--stdin")
    cmd.append(inp.pod)
    if inp.container:
        cmd += ["-c", inp.container]
    if inp.namespace:
        cmd += ["-n", inp.namespace]
    cmd += inp.additional_args
    cmd.append("--")
    cmd += inp.command
    return run_cli(cmd)


@mcp.tool(description="Start a kubectl port-forward session.")
def port_forward(inp: PortForwardInput) -> Dict[str, object]:
    resource_ref = f"{inp.resource}/{inp.name}" if "/" not in inp.name else inp.name
    cmd = ["kubectl", "port-forward", resource_ref] + inp.ports
    if inp.namespace:
        cmd += ["-n", inp.namespace]
    if inp.address:
        cmd += ["--address", inp.address]
    cmd += inp.additional_args

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
    except FileNotFoundError:
        return {"error": "kubectl executable not found on PATH.", "command": " ".join(cmd)}
    session_id = uuid.uuid4().hex
    with PORT_FORWARD_LOCK:
        PORT_FORWARD_PROCS[session_id] = proc
    return {
        "message": "Port-forward started.",
        "session_id": session_id,
        "pid": proc.pid,
        "command": " ".join(cmd),
        "note": "Use stop_port_forward(session_id) to terminate.",
    }


@mcp.tool(description="Stop an active port-forward session.")
def stop_port_forward(inp: PortForwardSessionInput) -> Dict[str, object]:
    with PORT_FORWARD_LOCK:
        proc = PORT_FORWARD_PROCS.pop(inp.session_id, None)
    if not proc:
        return {"error": f"No active port-forward session '{inp.session_id}'."}
    proc.terminate()
    try:
        proc.wait(timeout=5)
        stopped = True
    except subprocess.TimeoutExpired:
        proc.kill()
        stopped = False
    return {
        "message": "Port-forward terminated.",
        "session_id": inp.session_id,
        "pid": proc.pid,
        "graceful": stopped,
    }


@mcp.tool(description="Run 'kubectl top pods' for resource usage.")
def top_pods(inp: TopPodsInput) -> Dict[str, object]:
    cmd = ["kubectl", "top", "pods"]
    if inp.all_namespaces:
        cmd.append("-A")
    elif inp.namespace:
        cmd += ["-n", inp.namespace]
    if inp.no_headers:
        cmd.append("--no-headers")
    if inp.sort_by:
        cmd += ["--sort-by", inp.sort_by]
    return run_cli(cmd)


@mcp.tool(description="List all namespaces in the cluster.")
def list_namespaces() -> List[dict]:
    try:
        nss = core_v1.list_namespace().items
        return [
            {
                "name": ns.metadata.name,
                "status": ns.status.phase,
                "age": ns.metadata.creation_timestamp.isoformat(),
            }
            for ns in nss
        ]
    except ApiException as e:
        return [{"error": f"ERROR listing namespaces: {e}"}]


@mcp.tool(description="Manage kubectl contexts (current/list/use/set-namespace).")
def kubectl_context(inp: ContextInput) -> Dict[str, object]:
    if inp.action == "current":
        cmd = ["kubectl", "config", "current-context"]
    elif inp.action == "list":
        cmd = ["kubectl", "config", "get-contexts"]
    elif inp.action == "use":
        if not inp.context:
            return {"error": "Context name required for action 'use'."}
        cmd = ["kubectl", "config", "use-context", inp.context]
    elif inp.action == "set-namespace":
        if not inp.namespace:
            return {"error": "Namespace required for action 'set-namespace'."}
        cmd = ["kubectl", "config", "set-context", "--current", f"--namespace={inp.namespace}"]
    else:
        return {"error": f"Unsupported action '{inp.action}'."}
    cmd += inp.additional_args
    return run_cli(cmd)


@mcp.tool(description="List cluster nodes and basic conditions.")
def get_nodes() -> List[dict]:
    try:
        nodes = core_v1.list_node().items
        result = []
        for node in nodes:
            conditions = {cond.type: cond.status for cond in node.status.conditions or []}
            addresses = {addr.type: addr.address for addr in node.status.addresses or []}
            result.append(
                {
                    "name": node.metadata.name,
                    "labels": node.metadata.labels,
                    "taints": [t.to_dict() for t in node.spec.taints or []],
                    "conditions": conditions,
                    "capacity": node.status.capacity,
                    "allocatable": node.status.allocatable,
                    "addresses": addresses,
                    "age": node.metadata.creation_timestamp.isoformat(),
                }
            )
        return result
    except ApiException as e:
        return [{"error": f"ERROR listing nodes: {e}"}]


@mcp.tool(description="Fetch Kubernetes events (namespaced or cluster-wide).")
def events(inp: EventsInput) -> List[dict]:
    try:
        if inp.all_namespaces:
            evs = core_v1.list_event_for_all_namespaces(field_selector=inp.field_selector, limit=inp.limit).items
        else:
            namespace = inp.namespace or DEFAULT_NS
            evs = core_v1.list_namespaced_event(namespace=namespace, field_selector=inp.field_selector, limit=inp.limit).items
        return [
            {
                "type": ev.type,
                "reason": ev.reason,
                "message": ev.message,
                "involved_object": {
                    "kind": getattr(ev.involved_object, "kind", None),
                    "name": getattr(ev.involved_object, "name", None),
                    "namespace": getattr(ev.involved_object, "namespace", None),
                },
                "first_timestamp": getattr(ev, "first_timestamp", None).isoformat() if getattr(ev, "first_timestamp", None) else None,
                "last_timestamp": getattr(ev, "last_timestamp", None).isoformat() if getattr(ev, "last_timestamp", None) else None,
                "count": ev.count,
            }
            for ev in evs
        ]
    except ApiException as e:
        return [{"error": f"ERROR listing events: {e}"}]


@mcp.tool(description="Alias for OpenShift-like 'projects'; lists namespaces.")
def projects() -> List[dict]:
    return list_namespaces()


# ---------------------------
# Helm utilities
# ---------------------------
def _prepare_helm_command(base_cmd: List[str], additional_args: List[str]) -> List[str]:
    cmd = ["helm"] + base_cmd
    if additional_args:
        cmd += additional_args
    return cmd


@mcp.tool(description="Install or upgrade a Helm chart (helm upgrade --install).")
def install_helm_chart(inp: HelmInstallInput) -> Dict[str, object]:
    base_cmd = [
        "upgrade",
        "--install",
        inp.release_name,
        inp.chart,
        "--namespace",
        inp.namespace,
    ]
    if inp.create_namespace:
        base_cmd.append("--create-namespace")
    if inp.version:
        base_cmd += ["--version", inp.version]
    for key, value in inp.set.items():
        base_cmd += ["--set", f"{key}={value}"]

    cmd = _prepare_helm_command(base_cmd, inp.additional_args)
    if inp.values_yaml:
        cmd += ["-f", "-"]
        return run_cli(cmd, input_data=inp.values_yaml)
    return run_cli(cmd)


@mcp.tool(description="List Helm releases.")
def list_helm_releases(inp: HelmReleaseQuery) -> Dict[str, object]:
    base_cmd = ["list"]
    if inp.all_namespaces:
        base_cmd.append("-A")
    elif inp.namespace:
        base_cmd += ["-n", inp.namespace]
    if inp.filter:
        base_cmd += ["--filter", inp.filter]
    if inp.output:
        base_cmd += ["-o", inp.output]
    cmd = _prepare_helm_command(base_cmd, inp.additional_args)
    return run_cli(cmd)


@mcp.tool(description="Uninstall a Helm release.")
def uninstall_helm_release(inp: HelmUninstallInput) -> Dict[str, object]:
    base_cmd = ["uninstall", inp.release_name]
    if inp.namespace:
        base_cmd += ["-n", inp.namespace]
    if inp.keep_history:
        base_cmd.append("--keep-history")
    cmd = _prepare_helm_command(base_cmd, inp.additional_args)
    return run_cli(cmd)


@mcp.tool(description="Manage Helm repositories (add/update/remove/list).")
def helm_repo_management(inp: HelmRepoInput) -> Dict[str, object]:
    if inp.action == "add":
        if not inp.name or not inp.url:
            return {"error": "Both 'name' and 'url' are required for repo add."}
        base_cmd = ["repo", "add", inp.name, inp.url]
    elif inp.action == "update":
        base_cmd = ["repo", "update"]
    elif inp.action == "remove":
        if not inp.name:
            return {"error": "Repository name required for repo remove."}
        base_cmd = ["repo", "remove", inp.name]
    elif inp.action == "list":
        base_cmd = ["repo", "list"]
    else:
        return {"error": f"Unsupported action '{inp.action}'."}
    cmd = _prepare_helm_command(base_cmd, inp.additional_args)
    return run_cli(cmd)


if __name__ == "__main__":
    # Start the MCP server over HTTP
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=8080,
        path="/",
    )