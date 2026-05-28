import json

with open("train_cnn_bot.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Fix Cell 2: Add Simulator.java to the initial compile_cmd
for cell in nb["cells"]:
    src = "".join(cell["source"])
    if "src/student/DatasetGenerator.java" in src and "compile_cmd" in src:
        new_source = []
        for line in cell["source"]:
            if "src/student/DatasetGenerator.java" in line and "Simulator" not in line:
                line = line.replace(
                    'src/student/DatasetGenerator.java',
                    'src/student/DatasetGenerator.java\\", \\"src/student/Simulator.java'
                )
            new_source.append(line)
        cell["source"] = new_source
        break

# Fix Cell 7: Replace the recompile to include Simulator.java explicitly
for cell in nb["cells"]:
    src = "".join(cell["source"])
    if "Recompile updated StudentBotImpl" in src:
        cell["source"] = [
            "# Recompile the CNN-powered StudentBotImpl + Simulator\n",
            "print(\"Compiling Java CNN bot and simulator...\")\n",
            "recompile_cmd = [\n",
            "    \"javac\", \"-cp\", \"lib/arena-framework.jar:out\",\n",
            "    \"-d\", \"out\",\n",
            "    \"src/student/StudentBotImpl.java\", \"src/student/Simulator.java\"\n",
            "]\n",
            "subprocess.run(recompile_cmd, check=True)\n",
            "\n",
            "# Run evaluation\n",
            "print(\"Running evaluation on 300 matches...\")\n",
            "eval_cmd = [\n",
            "    \"java\", \"-cp\", \"lib/arena-framework.jar:out\",\n",
            "    \"student.Simulator\"\n",
            "]\n",
            "result = subprocess.run(eval_cmd, capture_output=True, text=True)\n",
            "print(result.stdout)\n",
            "if result.stderr:\n",
            "    print(\"Error:\", result.stderr)\n"
        ]
        break

with open("train_cnn_bot.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Simulator.java compile fix applied.")
