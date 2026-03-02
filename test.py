def matrix_mult(matrix_a, matrix_b):
    rows_a = len(matrix_a)
    cols_a = len(matrix_a[0])
    rows_b = len(matrix_b)
    cols_b = len(matrix_b[0])

    if cols_a != rows_b:
        raise ValueError("Matrix mult: cols_a must be equal rows_b")

    result = [[0] * cols_b for i in range(rows_a)]

    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    
    return result

def run_tests():
    test_cases = [
        {
            "name": "Square Matrix (2x2)",
            "A": [[1, 2], [3, 4]],
            "B": [[5, 6], [7, 8]],
            "expected": [[19, 22], [43, 50]]
        },
        {
            "name": "Rectangular (2x3 * 3x2)",
            "A": [[1, 2, 3], [4, 5, 6]],
            "B": [[7, 8], [9, 10], [11, 12]],
            "expected": [[58, 64], [139, 154]]
        }
    ]

    for case in test_cases:
        actual = matrix_mult(case["A"], case["B"])
        if actual == case["expected"]:
            print(f"✅ PASSED: {case['name']}")
        else:
            print(f"❌ FAILED: {case['name']}\n   Expected: {case['expected']}\n   Got:      {actual}")

    # Case 3: Error Handling
    print("\nTesting Error Handling (Incompatible Sizes):")
    try:
        invalid_a = [[1, 2]]  # 1x2
        invalid_b = [[1], [2], [3]]  # 3x1
        matrix_mult(invalid_a, invalid_b)
        print("❌ FAILED: Error should have been raised.")
    except ValueError as e:
        print(f"✅ PASSED: Caught expected error -> {e}")

if __name__ == "__main__":
    run_tests()
