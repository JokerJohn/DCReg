import os
# Suppress NumExpr warnings about thread count
os.environ.setdefault('NUMEXPR_MAX_THREADS', '8')

import numpy as np
import argparse
import sys
from pathlib import Path
import warnings

try:
    import pye57
except ImportError:
    print("Error: pye57 not installed. Install with: pip install pye57")
    sys.exit(1)

try:
    import open3d as o3d
except ImportError:
    print("Error: open3d not installed. Install with: pip install open3d")
    sys.exit(1)


def inspect_e57_file(e57_path):
    """
    Inspect the structure of an E57 file to understand available data

    Args:
        e57_path (str): Path to the E57 file
    """
    print(f"\n=== Inspecting E57 file: {e57_path} ===")

    try:
        e57 = pye57.E57(e57_path)
        print(f"Number of scans: {e57.scan_count}")

        for scan_index in range(min(e57.scan_count, 3)):  # Inspect first 3 scans
            print(f"\n--- Scan {scan_index + 1} ---")
            header = e57.get_header(scan_index)
            print(f"Point count: {header.point_count}")
            print(f"Fields in header: {header.point_fields}")

            # Try to read a small sample
            try:
                # Read all data (pye57 doesn't support partial reads)
                sample_data = e57.read_scan(scan_index, ignore_missing_fields=True)
                print(f"Actual fields in data: {list(sample_data.keys())}")

                # Check data types and shapes
                for field, data in list(sample_data.items())[:5]:  # First 5 fields
                    if hasattr(data, 'shape'):
                        print(f"  {field}: shape={data.shape}, dtype={data.dtype if hasattr(data, 'dtype') else 'N/A'}")
                    elif hasattr(data, '__len__'):
                        print(f"  {field}: length={len(data)}, type={type(data).__name__}")
                    else:
                        print(f"  {field}: type={type(data).__name__}")
            except Exception as e:
                print(f"Error reading scan data: {e}")

    except Exception as e:
        print(f"Error inspecting E57 file: {e}")


def read_e57_file_alternative(e57_path):
    """
    Alternative method to read E57 file with better field extraction

    Args:
        e57_path (str): Path to the E57 file

    Returns:
        tuple: (points, colors, intensities) where each is a numpy array or None
    """
    print(f"Using alternative E57 reader for: {e57_path}")

    e57 = pye57.E57(e57_path)
    imf = e57.image_file
    root = imf.root()

    if not root.isDefined("/data3D"):
        raise ValueError("No 3D data found in E57 file")

    data3d = root["/data3D"]

    all_points = []
    all_colors = []
    all_intensities = []

    for i in range(data3d.childCount()):
        print(f"\nProcessing scan {i + 1}/{data3d.childCount()}")
        scan = data3d[i]

        if not scan.isDefined("points"):
            print("  No points found in this scan")
            continue

        points_node = scan["points"]

        # Get prototype to understand structure
        prototype = points_node.prototype()
        if prototype:
            print(f"  Prototype fields: {[prototype[j].elementName() for j in range(prototype.childCount())]}")

        # Setup readers
        readers = {}

        # Try to setup readers for all possible fields
        field_mapping = {
            'cartesianX': 'x', 'cartesianY': 'y', 'cartesianZ': 'z',
            'colorRed': 'r', 'colorGreen': 'g', 'colorBlue': 'b',
            'intensity': 'intensity'
        }

        for e57_field, local_field in field_mapping.items():
            if points_node.isDefined(e57_field):
                readers[local_field] = []

        if not readers:
            print("  No readable fields found")
            continue

        # Create reader
        comp_reader = points_node.reader(readers)

        # Read data
        size = comp_reader.read()
        total_read = size

        while size > 0:
            size = comp_reader.read()
            total_read += size

        comp_reader.close()

        print(f"  Read {total_read} points from scan")

        # Convert to numpy arrays
        if all(k in readers for k in ['x', 'y', 'z']):
            points = np.column_stack((
                np.array(readers['x']),
                np.array(readers['y']),
                np.array(readers['z'])
            ))
            all_points.append(points)
            print(f"  Extracted {len(points)} coordinate values")

        if all(k in readers for k in ['r', 'g', 'b']):
            colors = np.column_stack((
                np.array(readers['r']),
                np.array(readers['g']),
                np.array(readers['b'])
            ))
            # Normalize if needed
            if colors.max() > 1.0:
                colors = colors / 255.0
            all_colors.append(colors)
            print(f"  Extracted {len(colors)} color values")

        if 'intensity' in readers:
            intensities = np.array(readers['intensity'])
            # Normalize if needed
            if intensities.max() > 1.0:
                intensities = intensities / intensities.max()
            all_intensities.append(intensities)
            print(f"  Extracted {len(intensities)} intensity values")

    # Combine all scans
    combined_points = np.vstack(all_points) if all_points else None
    combined_colors = np.vstack(all_colors) if all_colors else None
    combined_intensities = np.hstack(all_intensities) if all_intensities else None

    print(f"\nTotal points loaded: {len(combined_points) if combined_points is not None else 0}")
    if combined_colors is not None:
        print(f"Total color values: {len(combined_colors)}")
    if combined_intensities is not None:
        print(f"Total intensity values: {len(combined_intensities)}")

    # Print helpful message if fields were expected but not found
    if combined_points is not None and first_scan_header is not None:
        expected_colors = all(f in first_scan_header.point_fields for f in ['colorRed', 'colorGreen', 'colorBlue'])
        expected_intensity = 'intensity' in first_scan_header.point_fields

        if expected_colors and combined_colors is None:
            print("\nNote: Color fields were declared but could not be extracted.")
            if not force_fields:
                print("Try running with --force-fields flag to attempt alternative extraction methods.")

        if expected_intensity and combined_intensities is None:
            print("\nNote: Intensity field was declared but could not be extracted.")
            if not force_fields:
                print("Try running with --force-fields flag to attempt alternative extraction methods.")

    # If we didn't get color or intensity data but they were declared, try alternative method
    if force_fields and combined_points is not None and (combined_colors is None or combined_intensities is None):
        print("\nForce-fields option enabled: Trying alternative reading method...")
        try:
            alt_points, alt_colors, alt_intensities = read_e57_file_alternative(e57_path)
            if alt_colors is not None and combined_colors is None:
                combined_colors = alt_colors
                print("Successfully extracted colors with alternative method")
            if alt_intensities is not None and combined_intensities is None:
                combined_intensities = alt_intensities
                print("Successfully extracted intensities with alternative method")
        except Exception as e:
            print(f"Alternative method failed: {e}")

    return combined_points, combined_colors, combined_intensities


def read_e57_file(e57_path, force_fields=False):
    """
    Read point cloud data from an E57 file

    Args:
        e57_path (str): Path to the E57 file
        force_fields (bool): Whether to force extraction of all declared fields

    Returns:
        tuple: (points, colors, intensities) where each is a numpy array or None
    """
    print(f"Reading E57 file: {e57_path}")

    e57 = pye57.E57(e57_path)

    # Get the first scan (you can modify this to handle multiple scans)
    if e57.scan_count == 0:
        raise ValueError("No scans found in E57 file")

    print(f"Found {e57.scan_count} scan(s) in the file")

    # Store first scan header for later reference
    first_scan_header = e57.get_header(0) if e57.scan_count > 0 else None

    all_points = []
    all_colors = []
    all_intensities = []

    # Process each scan
    for scan_index in range(e57.scan_count):
        print(f"Processing scan {scan_index + 1}/{e57.scan_count}")

        # Get scan header to check available fields
        scan_header = e57.get_header(scan_index)

        # Print available fields for debugging
        print(f"Available fields in scan: {scan_header.point_fields}")

        # Read scan data using raw method for better field access
        try:
            # Try reading raw data which gives us structured arrays
            print("Reading scan data (this may take a moment for large files)...")

            # First try with ignore_missing_fields
            scan_data = e57.read_scan(scan_index, ignore_missing_fields=True)

            # Check if we got all expected fields
            expected_fields = ['cartesianX', 'cartesianY', 'cartesianZ']
            if 'intensity' in scan_header.point_fields:
                expected_fields.append('intensity')
            if all(f in scan_header.point_fields for f in ['colorRed', 'colorGreen', 'colorBlue']):
                expected_fields.extend(['colorRed', 'colorGreen', 'colorBlue'])

            missing_fields = [f for f in expected_fields if f not in scan_data]

            if missing_fields:
                print(f"Warning: Missing expected fields in scan data: {missing_fields}")

            if missing_fields and force_fields:
                print(f"Missing expected fields: {missing_fields}")
                print("Attempting to read with raw method...")

                # Try raw read
                raw_data = e57.read_scan_raw(scan_index)

                # Check if we got a structured array
                if isinstance(raw_data, np.ndarray) and raw_data.dtype.names:
                    print(f"Got structured array with fields: {list(raw_data.dtype.names)}")
                    scan_data = {}
                    for field in raw_data.dtype.names:
                        scan_data[field] = raw_data[field]
                elif isinstance(raw_data, dict):
                    scan_data = raw_data
                else:
                    print(f"Raw data type: {type(raw_data)}")

            # If still missing fields and force_fields is enabled, try manual extraction
            if missing_fields and force_fields and 'intensity' in missing_fields:
                print("\nAttempting manual intensity extraction...")
                try:
                    # Sometimes intensity is stored in a different way
                    # Try to access the raw E57 structure
                    imf = e57.image_file
                    root = imf.root()
                    if root.isDefined("/data3D"):
                        data3d = root["/data3D"]
                        if scan_index < data3d.childCount():
                            scan_node = data3d[scan_index]
                            if scan_node.isDefined("points"):
                                points_node = scan_node["points"]
                                if points_node.isDefined("intensity"):
                                    print("Found intensity field in E57 structure, but unable to extract with standard methods.")
                                    print("This E57 file may use a non-standard intensity format.")
                except Exception as e:
                    print(f"Manual extraction attempt failed: {e}")

        except Exception as e:
            print(f"Error reading scan: {e}")
            # Try basic read as fallback
            scan_data = e57.read_scan(scan_index, ignore_missing_fields=True)

        # Print actual fields in scan data for debugging
        if isinstance(scan_data, dict):
            print(f"Fields actually present in scan data: {list(scan_data.keys())}")

        # Check which fields are actually available in the data
        has_color = all([field in scan_data for field in ['colorRed', 'colorGreen', 'colorBlue']])

        # Check for different intensity field names in actual data
        intensity_field = None
        for field_name in ['intensity', 'rowIndex', 'columnIndex']:
            if field_name in scan_data:
                intensity_field = field_name
                break
        has_intensity = intensity_field is not None

        print(f"Has color data in scan: {has_color}")
        print(f"Has intensity data in scan: {has_intensity} (field: {intensity_field})")

        # Extract coordinates
        if all(coord in scan_data for coord in ['cartesianX', 'cartesianY', 'cartesianZ']):
            try:
                points = np.column_stack((
                    scan_data['cartesianX'],
                    scan_data['cartesianY'],
                    scan_data['cartesianZ']
                ))
                all_points.append(points)
                print(f"  Extracted {len(points)} points with Cartesian coordinates")
            except Exception as e:
                print(f"  Error extracting Cartesian coordinates: {e}")
        elif all(coord in scan_data for coord in ['sphericalRange', 'sphericalAzimuth', 'sphericalElevation']):
            # Convert spherical to Cartesian if needed
            try:
                print("  Converting spherical coordinates to Cartesian...")
                r = scan_data['sphericalRange']
                azimuth = scan_data['sphericalAzimuth']
                elevation = scan_data['sphericalElevation']

                # Convert to Cartesian
                x = r * np.cos(elevation) * np.cos(azimuth)
                y = r * np.cos(elevation) * np.sin(azimuth)
                z = r * np.sin(elevation)

                points = np.column_stack((x, y, z))
                all_points.append(points)
                print(f"  Extracted {len(points)} points from spherical coordinates")
            except Exception as e:
                print(f"  Error converting spherical coordinates: {e}")
        else:
            print("  Warning: No complete coordinate data found in scan")

        # Extract colors if available
        if has_color:
            try:
                colors = np.column_stack((
                    scan_data['colorRed'],
                    scan_data['colorGreen'],
                    scan_data['colorBlue']
                ))
                # Normalize to 0-1 range if needed
                if colors.max() > 1.0:
                    colors = colors / 255.0
                all_colors.append(colors)
                print(f"  Extracted {len(colors)} color values")
            except KeyError as e:
                print(f"  Warning: Color field {e} not found in scan data, skipping colors")
                has_color = False

        # Extract intensity if available
        if has_intensity:
            try:
                intensities = scan_data[intensity_field]
                # Normalize intensity if needed
                if intensities.max() > 1.0:
                    intensities = intensities / intensities.max()
                all_intensities.append(intensities)
                print(f"  Extracted {len(intensities)} intensity values")
            except KeyError as e:
                print(f"  Warning: Intensity field {e} not found in scan data, skipping intensity")
                has_intensity = False

    # Combine all scans
    combined_points = np.vstack(all_points) if all_points else None
    combined_colors = np.vstack(all_colors) if all_colors else None
    combined_intensities = np.hstack(all_intensities) if all_intensities else None

    print(f"\nTotal points loaded: {len(combined_points) if combined_points is not None else 0}")
    if combined_colors is not None:
        print(f"Total color values: {len(combined_colors)}")
    if combined_intensities is not None:
        print(f"Total intensity values: {len(combined_intensities)}")

    return combined_points, combined_colors, combined_intensities


def write_pcd_file(pcd_path, points, colors=None, intensities=None):
    """
    Write point cloud data to a PCD file using Open3D

    Args:
        pcd_path (str): Output path for the PCD file
        points (numpy.ndarray): Nx3 array of point coordinates
        colors (numpy.ndarray): Nx3 array of RGB colors (0-1 range)
        intensities (numpy.ndarray): N-length array of intensity values
    """
    print(f"Writing PCD file: {pcd_path}")

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Add colors if available
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # Note: Open3D doesn't directly support intensity in standard PCD format
    # If you need intensity, you might need to use a different library or custom implementation

    # Write to file
    success = o3d.io.write_point_cloud(pcd_path, pcd, write_ascii=False)

    if success:
        print(f"Successfully wrote {len(points)} points to {pcd_path}")
    else:
        raise IOError(f"Failed to write PCD file: {pcd_path}")

    return success


def write_pcd_with_color_intensity(pcd_path, points, colors=None, intensities=None):
    """
    Write PCD file with color and/or intensity fields (custom implementation)

    Args:
        pcd_path (str): Output path for the PCD file
        points (numpy.ndarray): Nx3 array of point coordinates
        colors (numpy.ndarray): Nx3 array of RGB colors (0-1 range)
        intensities (numpy.ndarray): N-length array of intensity values
    """
    print(f"Writing PCD file with custom format: {pcd_path}")

    num_points = len(points)

    # Determine fields based on available data
    fields = ["x", "y", "z"]
    sizes = ["4", "4", "4"]
    types = ["F", "F", "F"]
    counts = ["1", "1", "1"]

    if colors is not None and len(colors) == num_points:
        fields.extend(["r", "g", "b"])
        sizes.extend(["4", "4", "4"])
        types.extend(["F", "F", "F"])
        counts.extend(["1", "1", "1"])

    if intensities is not None and len(intensities) == num_points:
        fields.append("intensity")
        sizes.append("4")
        types.append("F")
        counts.append("1")

    # Write PCD header and data
    with open(pcd_path, 'w') as f:
        # Write header
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write(f"FIELDS {' '.join(fields)}\n")
        f.write(f"SIZE {' '.join(sizes)}\n")
        f.write(f"TYPE {' '.join(types)}\n")
        f.write(f"COUNT {' '.join(counts)}\n")
        f.write(f"WIDTH {num_points}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {num_points}\n")
        f.write("DATA ascii\n")

        # Write point data
        for i in range(num_points):
            line = f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}"

            if colors is not None and len(colors) == num_points:
                line += f" {colors[i, 0]:.6f} {colors[i, 1]:.6f} {colors[i, 2]:.6f}"

            if intensities is not None and len(intensities) == num_points:
                line += f" {intensities[i]:.6f}"

            f.write(line + "\n")

    print(f"Successfully wrote {num_points} points to {pcd_path}")
    if colors is not None:
        print(f"  - Included RGB color data")
    if intensities is not None:
        print(f"  - Included intensity data")


def convert_e57_to_pcd(e57_path, pcd_path, include_intensity=False, force_fields=False):
    """
    Main conversion function

    Args:
        e57_path (str): Path to input E57 file
        pcd_path (str): Path to output PCD file
        include_intensity (bool): Whether to include intensity data if available
        force_fields (bool): Whether to force extraction of color/intensity using alternative methods
    """
    try:
        # Read E57 file
        points, colors, intensities = read_e57_file(e57_path, force_fields)

        if points is None or len(points) == 0:
            raise ValueError("No point data found in E57 file")

        # Decide which writer to use based on available data
        has_extra_fields = (colors is not None) or (include_intensity and intensities is not None)

        if has_extra_fields:
            # Use custom writer for color and/or intensity
            write_pcd_with_color_intensity(pcd_path, points, colors,
                                           intensities if include_intensity else None)
        else:
            # Use Open3D writer for basic point clouds
            write_pcd_file(pcd_path, points, None)

        print("\nConversion completed successfully!")
        print(f"Output file: {pcd_path}")

    except Exception as e:
        print(f"\nError during conversion: {e}")
        print("\nRunning file inspection to help diagnose the issue...")
        inspect_e57_file(e57_path)
        raise


def batch_convert(input_dir, output_dir, include_intensity=False, force_fields=False):
    """
    Convert all E57 files in a directory

    Args:
        input_dir (str): Directory containing E57 files
        output_dir (str): Directory to save PCD files
        include_intensity (bool): Whether to include intensity data
        force_fields (bool): Whether to force extraction of color/intensity
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all E57 files
    e57_files = list(input_path.glob("*.e57"))

    if not e57_files:
        print(f"No E57 files found in {input_dir}")
        return

    print(f"Found {len(e57_files)} E57 file(s)")

    # Convert each file
    for i, e57_file in enumerate(e57_files):
        print(f"\nProcessing file {i+1}/{len(e57_files)}: {e57_file.name}")

        # Generate output filename
        pcd_file = output_path / e57_file.with_suffix('.pcd').name

        try:
            convert_e57_to_pcd(str(e57_file), str(pcd_file), include_intensity, force_fields)
        except Exception as e:
            print(f"Error converting {e57_file.name}: {str(e)}")
            continue

    print("\nBatch conversion completed!")


def main():
    parser = argparse.ArgumentParser(description="Convert E57 point cloud files to PCD format")
    parser.add_argument("input", help="Input E57 file or directory")
    parser.add_argument("-o", "--output", help="Output PCD file or directory (default: same name with .pcd extension)")
    parser.add_argument("-i", "--intensity", action="store_true", help="Include intensity data if available")
    parser.add_argument("-b", "--batch", action="store_true", help="Batch convert all E57 files in directory")
    parser.add_argument("--inspect", action="store_true", help="Inspect E57 file structure without converting")
    parser.add_argument("-f", "--force-fields", action="store_true", help="Force extraction of color/intensity fields using alternative methods")

    args = parser.parse_args()

    input_path = Path(args.input)

    # Handle inspection mode
    if args.inspect:
        if input_path.is_file() and input_path.suffix.lower() == '.e57':
            inspect_e57_file(str(input_path))
        else:
            print("Error: --inspect requires a single E57 file")
        sys.exit(0)

    if args.batch or input_path.is_dir():
        # Batch conversion
        if not input_path.is_dir():
            print("Error: Batch mode requires input to be a directory")
            sys.exit(1)

        output_dir = args.output if args.output else str(input_path)
        batch_convert(str(input_path), output_dir, args.intensity, args.force_fields)
    else:
        # Single file conversion
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            sys.exit(1)

        if not input_path.suffix.lower() == '.e57':
            print("Error: Input file must have .e57 extension")
            sys.exit(1)

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.with_suffix('.pcd')

        # Convert single file
        try:
            convert_e57_to_pcd(str(input_path), str(output_path), args.intensity, args.force_fields)
        except Exception as e:
            print(f"\nError during conversion: {str(e)}")
            print("\nTroubleshooting tips:")
            print("- If you see 'cartesianInvalidState' errors, the script should handle them automatically")
            print("- If intensity conversion fails, try without the -i flag")
            print("- Check that the e57 file is not corrupted")
            print("- Ensure you have enough memory for large point clouds")
            sys.exit(1)


if __name__ == "__main__":
    main()