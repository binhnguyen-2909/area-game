import json
import re

# 1. Update DatasetGenerator.java
with open("src/student/DatasetGenerator.java", "r") as f:
    java_code = f.read()

# Replace budget random logic
java_code = java_code.replace(
    "int b = rand.nextInt(7) + 10;",
    "int b = rand.nextInt(51); // Budget 0 to 50"
)

with open("src/student/DatasetGenerator.java", "w") as f:
    f.write(java_code)


# 2. Update Notebook
with open("train_cnn_bot.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    src = "".join(cell["source"])
    if "compile_cmd" in src and "DatasetGenerator" in src:
        new_source = []
        for line in cell["source"]:
            if "int b = rand.nextInt(7) + 10;" in line:
                line = line.replace("int b = rand.nextInt(7) + 10;", "int b = rand.nextInt(51); // Budget 0 to 50")
            new_source.append(line)
        cell["source"] = new_source

with open("train_cnn_bot.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Budget updated to 0-50 successfully.")
