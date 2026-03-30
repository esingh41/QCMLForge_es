import os.path as osp
from threading import Lock


_LMDB_ENV_REGISTRY = {}
_LMDB_ENV_LOCK = Lock()


def acquire_lmdb_env(
    lmdb_module,
    path,
    *,
    map_size,
    readonly,
    max_dbs=0,
    lock,
    max_readers=256,
):
    env_path = osp.abspath(path)

    with _LMDB_ENV_LOCK:
        entry = _LMDB_ENV_REGISTRY.get(env_path)
        if entry is not None:
            if entry["readonly"] and not readonly:
                raise RuntimeError(
                    f"The environment '{env_path}' is already open read-only in this process."
                )

            entry["refcount"] += 1
            return entry["env"]

        env = lmdb_module.open(
            env_path,
            map_size=map_size,
            readonly=readonly,
            max_dbs=max_dbs,
            lock=lock,
            max_readers=max_readers,
        )
        _LMDB_ENV_REGISTRY[env_path] = {
            "env": env,
            "readonly": readonly,
            "refcount": 1,
        }
        return env


def release_lmdb_env(path, env):
    if env is None:
        return

    env_path = osp.abspath(path)

    with _LMDB_ENV_LOCK:
        entry = _LMDB_ENV_REGISTRY.get(env_path)
        if entry is None or entry["env"] is not env:
            env.close()
            return

        entry["refcount"] -= 1
        if entry["refcount"] > 0:
            return

        _LMDB_ENV_REGISTRY.pop(env_path, None)
        env.close()
