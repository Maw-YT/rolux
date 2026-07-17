# models/

Model weights and TensorRT engines are **not** committed — `.onnx`, `.engine`,
`.npz`, and `.data` files are gitignored (large and/or GPU-specific).

Build engines with NVIDIA’s `trtexec` (see the root [README](../README.md#model-setup)).
Place the resulting `.engine` here and point the GUI at it.
