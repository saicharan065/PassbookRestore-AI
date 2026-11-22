# PassbookRestore AI: Evaluation Report

Documents processed: **24**  |  failures: **0**

## OCR accuracy (Tesseract, vs. ground truth)

| Metric | Baseline (raw scan) | Restored |
|---|---|---|
| Character error rate (lower is better) | 0.7926 | 0.2428 |
| Word accuracy (higher is better) | 0.301 | 0.598 |
| Mean OCR confidence | 48.8875 | 65.5 |

CER improved on **24 of 24** documents.

## Image quality

| Metric | Original | Restored |
|---|---|---|
| Sharpness (Laplacian variance @1000px) | 240.1725 | 266.5513 |
| Contrast (P95-P5) | 70.9167 | 158.1667 |
| Background std (lower = cleaner paper) | 3.5596 | 2.3696 |
| Noise sigma | 1.955 | 0.5913 |
| SSIM vs clean reference | 0.6169 | 0.7715 |

## Structure preservation

Score combines ruled-line counts, text-row counts, aspect ratio and ink overlap against the reference layout (1.0 = identical).

Mean structure score: scan **0.8307** -> restored **0.966**.

## By document type

| Type | n | CER base | CER restored | Word acc base | Word acc restored | Structure |
|---|---|---|---|---|---|---|
| cheque | 8 | 0.5022 | 0.2492 | 0.525 | 0.575 | 0.9597 |
| kyc | 8 | 1.0495 | 0.1958 | 0.2742 | 0.7379 | 0.9805 |
| passbook | 8 | 0.826 | 0.2833 | 0.1037 | 0.4811 | 0.9576 |

## Per-document results

| File | Type | CER base | CER restored | Struct | Upscaler | Time (s) |
|---|---|---|---|---|---|---|
| cheque_01.png | cheque | 0.058 | 0.0435 | 0.964 | opencv-lanczos | 1.31 |
| cheque_02.png | cheque | 0.0725 | 0.0072 | 1.0 | opencv-lanczos | 1.29 |
| cheque_03.png | cheque | 0.2817 | 0.0141 | 0.984 | opencv-lanczos | 1.6 |
| cheque_04.png | cheque | 0.2847 | 0.0903 | 0.999 | opencv-lanczos | 1.51 |
| cheque_05.png | cheque | 0.4795 | 0.137 | 0.88 | opencv-lanczos | 1.43 |
| cheque_06.png | cheque | 0.7042 | 0.2817 | 0.964 | opencv-lanczos | 1.33 |
| cheque_07.png | cheque | 1.2929 | 0.6714 | 0.999 | opencv-lanczos | 1.33 |
| cheque_08.png | cheque | 0.8444 | 0.7481 | 0.888 | opencv-lanczos | 1.49 |
| kyc_01.png | kyc | 0.665 | 0.069 | 1.0 | opencv-lanczos | 1.11 |
| kyc_02.png | kyc | 0.665 | 0.069 | 1.0 | opencv-lanczos | 1.2 |
| kyc_03.png | kyc | 0.5922 | 0.068 | 1.0 | opencv-lanczos | 1.01 |
| kyc_04.png | kyc | 0.4485 | 0.0722 | 1.0 | opencv-lanczos | 1.22 |
| kyc_05.png | kyc | 0.6515 | 0.101 | 0.999 | opencv-lanczos | 1.08 |
| kyc_06.png | kyc | 0.8333 | 0.1961 | 0.999 | opencv-lanczos | 1.11 |
| kyc_07.png | kyc | 1.8252 | 0.2864 | 0.959 | opencv-lanczos | 1.68 |
| kyc_08.png | kyc | 2.715 | 0.705 | 0.887 | opencv-lanczos | 1.87 |
| passbook_01.png | passbook | 0.7571 | 0.0355 | 0.984 | opencv-lanczos | 2.11 |
| passbook_02.png | passbook | 0.653 | 0.0323 | 1.0 | opencv-lanczos | 1.77 |
| passbook_03.png | passbook | 0.7512 | 0.056 | 0.997 | opencv-lanczos | 1.81 |
| passbook_04.png | passbook | 0.8199 | 0.1184 | 0.99 | opencv-lanczos | 1.83 |
| passbook_05.png | passbook | 0.878 | 0.1548 | 0.982 | opencv-lanczos | 1.31 |
| passbook_06.png | passbook | 1.0 | 0.4014 | 0.945 | opencv-lanczos | 1.47 |
| passbook_07.png | passbook | 0.9219 | 0.699 | 0.896 | opencv-lanczos | 1.47 |
| passbook_08.png | passbook | 0.8272 | 0.7689 | 0.867 | opencv-lanczos | 2.33 |
