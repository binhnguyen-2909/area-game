import json

with open("train_cnn_bot.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Change classpath separators from ';' to ':' for Linux
def fix_classpath(cells):
    for c in cells:
        new_source = []
        for line in c["source"]:
            # replace Windows classpath with Linux classpath
            new_line = line.replace("lib/arena-framework.jar;out", "lib/arena-framework.jar:out")
            new_source.append(new_line)
        c["source"] = new_source

fix_classpath(nb["cells"])

# Add explicit chdir to server path
def find_cell(query):
    for i, c in enumerate(nb["cells"]):
        src = "".join(c["source"])
        if query in src:
            return i
    return -1

idx_setup = find_cell("import os")
if idx_setup != -1:
    source = nb["cells"][idx_setup]["source"]
    new_source = []
    for line in source:
        new_source.append(line)
        if "import os\n" == line:
            new_source.append("\n# Update to Server Path if running on Linux Server\n")
            new_source.append("SERVER_PATH = \"/home/hvusynh2/nguyenduong/arena-student-kit 4/arena-student-kit\"\n")
            new_source.append("import sys\n")
            new_source.append("if os.path.exists(SERVER_PATH):\n")
            new_source.append("    os.chdir(SERVER_PATH)\n")
            new_source.append("    print(\"Changed directory to server path:\", SERVER_PATH)\n")
    nb["cells"][idx_setup]["source"] = new_source

with open("train_cnn_bot.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Linux paths updated successfully.")
