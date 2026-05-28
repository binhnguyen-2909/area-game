import json

with open("train_cnn_bot.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Fix Cell 2: Replace the broken compile line
for cell in nb["cells"]:
    src = "".join(cell["source"])
    if "compile_cmd" in src and "DatasetGenerator" in src:
        cell["source"] = [
            "# Compile the Java bot, dataset generator, and simulator\n",
            "print(\"Compiling Java files...\")\n",
            "compile_cmd = [\n",
            "    \"javac\", \"-cp\", \"lib/arena-framework.jar:out\",\n",
            "    \"-d\", \"out\",\n",
            "    \"src/student/StudentBotImpl.java\", \"src/student/DatasetGenerator.java\", \"src/student/Simulator.java\"\n",
            "]\n",
            "subprocess.run(compile_cmd, check=True)\n",
            "\n",
            "# Run the dataset generator\n",
            "print(\"Running dataset generator to collect expert moves...\")\n",
            "run_cmd = [\n",
            "    \"java\", \"-cp\", \"lib/arena-framework.jar:out\",\n",
            "    \"student.DatasetGenerator\"\n",
            "]\n",
            "subprocess.run(run_cmd, check=True)\n"
        ]
        break

with open("train_cnn_bot.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Compile cell fixed properly.")
