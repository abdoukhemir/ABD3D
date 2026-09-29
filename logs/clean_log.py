import re

INPUT_FILE = "training.log"
OUTPUT_FILE = "clean_steps.log"

pattern = re.compile(
    r"Step \d+/\d+ \| Loss: .*"
)

with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as infile, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    for line in infile:
        line = line.strip()

        # Find "Step ..." anywhere in the line
        match = pattern.search(line)

        if match:
            # Write only the Step part, removing timestamps/prefixes
            outfile.write(match.group(0) + "\n")

print(f"Done! Clean log saved to: {OUTPUT_FILE}")