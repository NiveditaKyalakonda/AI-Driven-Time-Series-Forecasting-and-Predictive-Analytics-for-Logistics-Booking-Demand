
from pathlib import Path
import posixpath
import re
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
_BOOKING_DATE_CELL = re.compile(
    rb'<c\s+r="C(\d+)"[^>]*>\s*<v>([^<]+)</v>\s*</c>'
)


def get_excel_sheets(file_name="BK_AUG2026.xlsx"):
    """Return the sheet names in an Excel workbook."""
    file_path = RAW_DATA_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"Excel file not found: {file_path}")

    return pd.ExcelFile(file_path).sheet_names


def load_excel(file_name="BK_AUG2026.xlsx", sheet_name=None, usecols=None):
    """Load one sheet, or combine all workbook sheets when sheet_name is None."""
    file_path = RAW_DATA_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            "Check the filename and data/raw folder."
        )

    sheets = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        usecols=usecols
    )

    if isinstance(sheets, dict):
        frames = []
        for name, frame in sheets.items():
            frame["SOURCE_SHEET"] = name
            frames.append(frame)
        df = pd.concat(frames, ignore_index=True)
    else:
        df = sheets

    df.columns = [str(col).strip() for col in df.columns]

    if df.empty:
        raise ValueError("The selected Excel sheet is empty.")

    print("Dataset loaded successfully!")
    print("File:", file_path)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    return df


def load_booking_date_counts(file_name="BK_AUG2026.xlsx", expected_month=None):
    """Count booking rows by date while streaming only date cells."""
    file_path = RAW_DATA_DIR / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"Excel file not found: {file_path}")

    main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    package_rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    serial_counts = {}

    with ZipFile(file_path) as workbook:
        workbook_root = ET.fromstring(workbook.read("xl/workbook.xml"))
        workbook_properties = workbook_root.find(
            f"{{{main_ns}}}workbookPr"
        )
        date_origin = (
            "1904-01-01"
            if workbook_properties is not None
            and workbook_properties.attrib.get("date1904") in {"1", "true"}
            else "1899-12-30"
        )
        relationships = ET.fromstring(
            workbook.read("xl/_rels/workbook.xml.rels")
        )
        targets = {
            item.attrib["Id"]: item.attrib["Target"]
            for item in relationships.findall(
                f"{{{package_rel_ns}}}Relationship"
            )
        }
        sheets = workbook_root.findall(
            f".//{{{main_ns}}}sheet"
        )
        if not sheets:
            raise ValueError(f"No worksheets found in {file_path}.")

        for sheet in sheets:
            relation_id = sheet.attrib[f"{{{rel_ns}}}id"]
            target = targets.get(relation_id)
            if target is None:
                raise ValueError(
                    f"Worksheet relationship not found for {sheet.attrib['name']}."
                )
            worksheet_path = target.lstrip("/")
            if not worksheet_path.startswith("xl/"):
                worksheet_path = posixpath.normpath(
                    posixpath.join("xl", worksheet_path)
                )

            with workbook.open(worksheet_path) as source:
                remainder = b""
                while chunk := source.read(1024 * 1024):
                    buffer = remainder + chunk
                    process_limit = max(0, len(buffer) - 512)
                    first_unprocessed = None
                    for match in _BOOKING_DATE_CELL.finditer(buffer):
                        if match.end() > process_limit:
                            first_unprocessed = match.start()
                            break
                        if match.group(1) == b"1":
                            continue
                        try:
                            serial = float(match.group(2))
                        except ValueError:
                            continue
                        serial_counts[serial] = (
                            serial_counts.get(serial, 0) + 1
                        )
                    if first_unprocessed is None:
                        last_date_cell = buffer.rfind(
                            b'<c r="C', 0, process_limit
                        )
                        last_closed_cell = buffer.rfind(
                            b"</c>", 0, process_limit
                        )
                        remainder_start = (
                            last_date_cell
                            if last_date_cell > last_closed_cell
                            else process_limit
                        )
                        remainder = buffer[remainder_start:]
                    else:
                        remainder = buffer[first_unprocessed:]

                for match in _BOOKING_DATE_CELL.finditer(remainder):
                    if match.group(1) == b"1":
                        continue
                    try:
                        serial = float(match.group(2))
                    except ValueError:
                        continue
                    serial_counts[serial] = serial_counts.get(serial, 0) + 1

    counts = {}
    for serial, count in serial_counts.items():
        date_key = pd.to_datetime(
            serial,
            unit="D",
            origin=date_origin
        ).normalize()
        counts[date_key] = counts.get(date_key, 0) + count

    if expected_month is not None:
        expected_period = pd.Period(expected_month, freq="M")
        invalid_rows = sum(
            count for date, count in counts.items()
            if date.to_period("M") != expected_period
        )
        if invalid_rows:
            print(
                f"Excluded {invalid_rows} booking rows outside "
                f"the expected period {expected_period}."
            )
        counts = {
            date: count for date, count in counts.items()
            if date.to_period("M") == expected_period
        }

    if not counts:
        raise ValueError(
            f"No booking dates were found in {file_path}. "
            "Expected dates in the BKDATE column (column C) and selected "
            "period."
        )

    return (
        pd.DataFrame(
            sorted(counts.items()),
            columns=["DATE", "BOOKING_COUNT"]
        )
    )


if __name__ == "__main__":
    print("Excel sheets:", get_excel_sheets())
    data = load_excel()
    print(data.head())
