# Word report source

This directory contains the report authoring code used for the logits flight recorder delivery: content and illustrations, fixed-width tables, and explicit Windows font settings.

Download flight-recorder-report-assets.zip from the [delivery Release](https://github.com/wanghuanjun2113/vllm-ascend/releases/tag/logits-flight-recorder-20260907) and extract it into a writable report directory. The archive supplies assets/ with four technical illustrations, two real HTML screenshots, prompts and checksums. No private reference document is needed; page measurements and table formatting are encoded in fix_format.py.

With Python 3 and python-docx installed, run from the repository root:

    export FLIGHT_REPORT_DIR=/absolute/path/to/report-directory
    python tools/flight_recorder/report/create_report.py
    python tools/flight_recorder/report/fix_format.py
    python tools/flight_recorder/report/windows_fonts.py

The final file is $FLIGHT_REPORT_DIR/logits-design-windows.docx. The first two files are build intermediates. Font names are Microsoft YaHei for Chinese, Arial for Latin text, and Consolas for code. Rendering on Linux uses available substitute fonts; this is not evidence of Windows-native visual verification.

The scripts verify consistent table/grid/cell widths, unchanged text during formatting, and unchanged embedded images during font changes. Validate the resulting document visually with your Word renderer before redistributing an edited report. This report describes the measured 2026-09-07 workload and fixed software revisions; rerunning the authoring scripts does not rerun model benchmarks.
