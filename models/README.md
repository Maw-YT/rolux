# models/

Model weights and TensorRT engines are **not** committed — `.onnx`, `.engine`,
`.npz`, and `.data` files are gitignored (large and/or GPU-specific).

Build the depth engine here:

```bash
python export_trt.py --onnx models/depth_anything_v2_vits_fp16.onnx --height 392 --width 392
```

The app looks for `models/depth_anything_v2_vits_fp16.engine` by default (you can
point the GUI at any `.engine`). See the root `README.md` for full instructions.
