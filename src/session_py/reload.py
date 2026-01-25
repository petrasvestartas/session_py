"""Generic hot-reload for any Python package without restarting Rhino."""
import sys
import importlib
from pathlib import Path


def reload_package(package_name, verbose=True):
    """Reload all modules of a package in dependency order.

    Works with any package, not just session_py.

    Usage:
        from session_py.reload import reload_package
        reload_package("session_py")
        reload_package("my_other_lib")
        reload_package("compas")

    Args:
        package_name: Root package name (e.g., "session_py", "compas")
        verbose: Print reload progress

    Returns:
        List of reloaded module names
    """
    pkg_modules = [
        name for name in sys.modules.keys()
        if name == package_name or name.startswith(package_name + ".")
    ]

    if not pkg_modules:
        if verbose:
            print(f"No modules loaded for '{package_name}'")
        return []

    pkg_modules.sort(key=lambda x: x.count('.'), reverse=True)

    reloaded = []
    failed = []

    for mod_name in pkg_modules:
        module = sys.modules.get(mod_name)
        if module is None:
            continue
        try:
            importlib.reload(module)
            reloaded.append(mod_name)
            if verbose:
                print(f"  + {mod_name}")
        except Exception as e:
            failed.append((mod_name, str(e)))
            if verbose:
                print(f"  x {mod_name}: {e}")

    if verbose:
        print(f"Reloaded {len(reloaded)}/{len(pkg_modules)} modules")
        if failed:
            print(f"Failed: {len(failed)}")

    return reloaded


def clear_package(package_name, verbose=True):
    """Remove all modules of a package from cache (nuclear option).

    After calling this, you must re-import the package.

    Usage:
        from session_py.reload import clear_package
        clear_package("session_py")
        from session_py import Point  # must re-import
    """
    removed = []
    for mod_name in list(sys.modules.keys()):
        if mod_name == package_name or mod_name.startswith(package_name + "."):
            del sys.modules[mod_name]
            removed.append(mod_name)

    if verbose:
        print(f"Cleared {len(removed)} modules from '{package_name}'")

    return removed


def reload_file(filepath, verbose=True):
    """Reload a single .py file if it's loaded as a module.

    Usage:
        reload_file("C:/path/to/my_module.py")
    """
    filepath = Path(filepath).resolve()

    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        try:
            mod_file = getattr(module, '__file__', None)
            if mod_file and Path(mod_file).resolve() == filepath:
                importlib.reload(module)
                if verbose:
                    print(f"Reloaded: {name}")
                return name
        except Exception:
            pass

    if verbose:
        print(f"Module not found for: {filepath}")
    return None


def reload_session():
    """Shortcut for reload_package('session_py')."""
    return reload_package("session_py")


def clear_session():
    """Shortcut for clear_package('session_py')."""
    return clear_package("session_py")
