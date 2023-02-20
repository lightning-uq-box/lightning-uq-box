"""ReforesTree Dataset."""

from typing import Callable, Dict, Optional

from torch import Tensor
from torchgeo.datasets import ReforesTree


class ReforesTreeRegression(ReforesTree):
    """ReforesTree Dataset focusing on Regresion Task.

    Wrapper around TorchGeo implementation.
    """

    def __init__(
        self,
        root: str = "data",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new instance of Dataset."""
        super().__init__(root, transforms, download, checksum)
