"""Current model, its controlled ablations, and independent comparators."""

from Baselines.models import KANHAR, KANMLPMixer, ResidualKANFeatures, TemporalCNN
from Proposed_Method.model import GeometryKAN


GEOMETRY = {
    "geometry_kan": (True, True, True),
    "geometry_axes": (False, True, True),
    "geometry_mlp": (True, False, True),
    "geometry_no_posture": (True, True, False),
    "geometry_no_rotation": (True, True, True),
}
PRIMARY = "geometry_kan"
ABLATIONS = tuple(name for name in GEOMETRY if name != PRIMARY)
BASELINES = ("temporal_cnn", "kan_har", "kan_fe", "kan_mixer")
MODELS = (*GEOMETRY, *BASELINES)
NAMES = {
    "geometry_kan": "Geometry TempoKAN",
    "geometry_axes": "Original-axis features",
    "geometry_mlp": "Geometry · MLP readouts",
    "geometry_no_posture": "Without posture head",
    "geometry_no_rotation": "Without mounting augmentation",
    "temporal_cnn": "Temporal CNN",
    "kan_har": "KAN-HAR",
    "kan_fe": "RL-KAN feature extractor",
    "kan_mixer": "KAN–MLP–Mixer",
    "geometry_average": "Geometry KAN + MLP average",
}
BASELINE_WIDTHS = {"temporal_cnn": 48, "kan_har": 20, "kan_fe": 24, "kan_mixer": 32}
FOLDS = {"uci_har": 1, "pamap2": 8, "selfback": 5}
TASKS = (("uci_har", "clean"), ("pamap2", "clean"), ("selfback", "clean"), ("selfback", "pace"))
DATA_NAMES = {"uci_har": "UCI HAR", "pamap2": "PAMAP2", "selfback": "SELFBACK"}


def make_model(name, channels, classes, width=24, stride=4):
    if name in GEOMETRY:
        aligned, spline, posture = GEOMETRY[name]
        return GeometryKAN(channels, classes, width, stride, aligned=aligned, spline=spline, posture=posture)
    if name == "temporal_cnn":
        return TemporalCNN(channels, classes, width, stride)
    if name == "kan_har":
        return KANHAR(channels, classes, width)
    if name == "kan_fe":
        return ResidualKANFeatures(channels, classes)
    if name == "kan_mixer":
        return KANMLPMixer(channels, classes, width)
    raise ValueError(f"Unknown model {name}")


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
