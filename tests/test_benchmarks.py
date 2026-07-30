# Import packages:
import importlib


# The test modules to run, in order:
TESTS = [
    ("newton", "tests.test_newton"),
    ("lambda", "tests.test_lambda"),
    ("obstacles", "tests.test_obstacles"),
    ("lqr", "tests.test_lqr"),
    ("reachability", "tests.test_reachability"),
    ("optimizer", "tests.test_optimizer"),
    ("codesign", "tests.test_codesign"),
]

# Run every test module and print the results:
def main():

    # Header:
    print("=" * 74)
    print("GRACE TEST SUITE")
    print("=" * 74)

    # Track the overall pass count:
    total = 0
    passed = 0

    # Run each test module:
    for group, module_name in TESTS:

        # Import and run the module:
        print(f"\n[{group}]")
        module = importlib.import_module(module_name)
        records = module.run()

        # Print each record:
        for name, ok, detail in records:
            total += 1
            passed += 1 if ok else 0
            tag = "PASS" if ok else "FAIL"
            print(f"  [{tag}] {name}: {detail}")

    # Summary:
    print("\n" + "=" * 74)
    print(f"RESULT: {passed}/{total} passed")
    print("=" * 74)

    # Return the overall success:
    return passed == total


# Run when executed as a script:
if __name__ == "__main__":
    main()
