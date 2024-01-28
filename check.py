def compare_files_with_tolerance(file1, file2, tolerance=1e-6):
    try:
        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            for line_num, (line1, line2) in enumerate(zip(f1, f2), 1):
                # Convert line strings to lists of floats
                entries1 = list(map(float, line1.strip().split()))
                entries2 = list(map(float, line2.strip().split()))

                # Check if the lengths of the lines are different
                if len(entries1) != len(entries2):
                    return f"Files differ in number of columns at line {line_num}."

                # Compare the values with the specified tolerance
                for val1, val2 in zip(entries1, entries2):
                    if abs(val1 - val2) > tolerance:
                        return f"Files differ at line {line_num} with values {val1} and {val2}."

            return "Files are the same within the specified tolerance."
    except FileNotFoundError as e:
        return f"File not found: {e}"

# Replace 'output.txt' and 'output2.txt' with your actual file paths if needed
result = compare_files_with_tolerance('output.txt', 'output2.txt')
print(result)
