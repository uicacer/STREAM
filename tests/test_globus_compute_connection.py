from globus_compute_sdk import Executor

endpoint_id = "8d978809-eec4-413d-bbd4-b099e488100a"


def test_lakeshore():
    """Test with imports - should work now with matching Python versions"""
    import os
    import platform
    import socket

    return {
        "hostname": socket.gethostname(),
        "user": os.environ.get("USER", "unknown"),
        "python": platform.python_version(),
        "node": platform.node(),
    }


if __name__ == "__main__":
    print(f"Connecting to endpoint: {endpoint_id}")

    with Executor(endpoint_id=endpoint_id) as gce:
        future = gce.submit(test_lakeshore)
        print("Task submitted, waiting for result...")
        try:
            result = future.result(timeout=120)
            print("\n✅ SUCCESS!")
            print(f"Hostname: {result['hostname']}")
            print(f"User: {result['user']}")
            print(f"Python: {result['python']}")
        except Exception as e:
            print(f"❌ ERROR: {e}")
        print(f"Node: {result['node']}")
