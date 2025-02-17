import os
import re


base_path = r'.'
file_paths = [
    r'models/__init__.py',
    r'models/SFANet.py',
    r'models/VGG.py',
    
    r'scripts/__init__.py',
    r'scripts/calculate_ap.py',
    r'scripts/inference.py',
    r'scripts/prepare.py',
    r'scripts/test.py',
    r'scripts/train.py',
    r'scripts/tune.py',
    
    r'utils/__init__.py',
    r'utils/evaluate.py',
    r'utils/inference.py',
    r'utils/preprocess.py',
    
    r'environment.yml',
    r'LICENSE',
    r'README.md'
]


def aggregate_files(base_path, relative_paths):
    aggregated_content = ""
    for relative_path in relative_paths:
        full_path = os.path.join(base_path, relative_path)
        aggregated_content += f"=== {full_path} ===\n"
        try:
            with open(full_path, 'r', encoding='utf-8') as file:
                content = file.read()
                aggregated_content += content + "\n\n"
        except FileNotFoundError:
            raise Exception(f"Error: File not found: {full_path}\n\n")
        except Exception as e:
            raise Exception(f"Error reading {full_path}: {e}\n\n")
    return aggregated_content


def replace_import_groups(content):
    pattern = r'(?m)^(import\s+.*\n)+'
    modified_content = re.sub(pattern, '...\n', content)
    return modified_content


def main():
    aggregated_string = aggregate_files(base_path, file_paths)
    #aggregated_string = replace_import_groups(aggregated_string)
    output_path = os.path.join(base_path, 'aggregated_contents.txt')
    try:
        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write(aggregated_string)
        print(f"Aggregated content successfully saved to {output_path}")
    except Exception as e:
        raise Exception(f"Failed to write aggregated content to file: {e}")


if __name__ == "__main__":
    main()