# Third-Party Licenses

This repository is distributed as a whole under the **GNU General Public License v3.0**
(see [`LICENSE`](LICENSE)). GPL-3.0 does not extinguish the license terms of the
third-party components it incorporates; this file preserves those terms and
identifies the files they apply to, per MIT License condition 2 ("the above
copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software") and GPL-3.0 §5(b)/§5(c).

---

## SSARCandy/Coherent-Line-Drawing — MIT License

- **Source:** https://github.com/SSARCandy/Coherent-Line-Drawing
- **Original license:** MIT
- **Role in this project:** Coherent Line Drawing (CLD) and Flow-based Difference
  of Gaussians (FDoG) implementation, based on Kang et al., *"Coherent Line
  Drawing"*, Proc. NPAR 2007.
- **Note:** the version integrated here predates upstream v1.0.3 (still includes
  `postProcessing.{h,cpp}`, later removed upstream).

**Files covered by this notice:**

```
include/CLD.h
include/ETF.h
include/const.h
include/gui.h
include/postProcessing.h
src/CLD.cpp
src/ETF.cpp
src/cmd.cpp
src/gui.cpp
src/main.cpp
src/postProcessing.cpp
```

**Original license text:**

```
MIT License

Copyright (c) 2019 HSU,SHU-HSUAN

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Code-Break0/Image-Processing — GPL-3.0

- **Source:** https://github.com/Code-Break0/Image-Processing
- **Original license:** GNU General Public License v3.0
- **Role in this project:** original architecture and base concepts for image
  data processing.

This component's license is identical to the license governing this repository
as a whole; its full text is at [`LICENSE`](LICENSE). If the upstream project's
own `LICENSE` file or source headers contain a filled-in copyright line
(`Copyright (C) <year> <author>`, as opposed to the unfilled FSF boilerplate),
that line should be reproduced verbatim here — **TODO: verify against upstream
and complete this section.**

---

## Modifications

Files carrying substantive modifications from their upstream originals should
state so in a header comment at the top of the file, per GPL-3.0 §5(a)
("carry prominent notices stating that you changed the files and the date of
any change"). This is a per-file source requirement, not satisfied by this
document or by the README alone.
