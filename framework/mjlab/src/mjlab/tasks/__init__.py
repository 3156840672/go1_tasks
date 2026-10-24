from importlib.metadata import entry_points

from mjlab.utils.lab_api.tasks.importer import import_packages

_BLACKLIST_PKGS = ["utils", ".mdp"]

import_packages(__name__, _BLACKLIST_PKGS)

# Load task packages installed outside the bundled mjlab source tree.
for task_entry_point in entry_points(group="mjlab.tasks"):
  task_entry_point.load()
