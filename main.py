import sys
from classifier import classify_provision, classify_provision_with_file_search

matrix_path = "data/cso-matrix.txt"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py \"Your provision text here\"")
        sys.exit(1)

    provision_text = sys.argv[1]
    ###Uncomment this line with # to use the original classifier, which uses the matrix directly
    result = classify_provision(provision_text)

    #Uses the file search tool to classify the provision
    #result = classify_provision_with_file_search(provision_text, matrix_path)

    print("Classification Result:\n", result)