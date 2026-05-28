import json

with open("train_cnn_bot.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

def find_cell(query):
    for i, c in enumerate(nb["cells"]):
        src = "".join(c["source"])
        if query in src:
            return i
    return -1

# FIX 1: Remove unused pandas import
idx_setup = find_cell("import pandas as pd")
if idx_setup != -1:
    nb["cells"][idx_setup]["source"] = [
        line for line in nb["cells"][idx_setup]["source"]
        if "import pandas as pd" not in line
    ]

# FIX 2: ByteOrder.nativeOrder() -> ByteOrder.LITTLE_ENDIAN
idx_export = find_cell("ByteOrder.nativeOrder()")
if idx_export != -1:
    new_source = []
    for line in nb["cells"][idx_export]["source"]:
        new_source.append(line.replace("ByteOrder.nativeOrder()", "ByteOrder.LITTLE_ENDIAN"))
    nb["cells"][idx_export]["source"] = new_source

# FIX 3: Add Simulator.java to compile_cmd for Cell 2 (dataset generation)
idx_compile = find_cell("src/student/StudentBotImpl.java\\\", \\\"src/student/DatasetGenerator.java")
if idx_compile != -1:
    new_source = []
    for line in nb["cells"][idx_compile]["source"]:
        line = line.replace(
            '"src/student/StudentBotImpl.java\\", \\"src/student/DatasetGenerator.java\\"',
            '"src/student/StudentBotImpl.java\\", \\"src/student/DatasetGenerator.java\\", \\"src/student/Simulator.java\\"'
        )
        new_source.append(line)
    nb["cells"][idx_compile]["source"] = new_source

# FIX 4: Multi-epoch training (re-stream file multiple times)
idx_train = find_cell("Training the model on GPU (Streaming 1 Epoch)")
if idx_train != -1:
    nb["cells"][idx_train]["source"] = [
        "train_loader = DataLoader(dataset, batch_size=256)\n",
        "\n",
        "criterion = nn.CrossEntropyLoss()\n",
        "optimizer = optim.Adam(model.parameters(), lr=0.003)\n",
        "\n",
        "NUM_EPOCHS = 5\n",
        "print(f\"Training the model on GPU ({NUM_EPOCHS} epochs, streaming from disk)...\")\n",
        "\n",
        "for epoch in range(NUM_EPOCHS):\n",
        "    model.train()\n",
        "    total_loss = 0.0\n",
        "    correct = 0\n",
        "    total = 0\n",
        "    step = 0\n",
        "    \n",
        "    # Re-create dataset to re-stream from beginning of file\n",
        "    epoch_dataset = StreamingArenaDataset(\"dataset.csv\")\n",
        "    epoch_loader = DataLoader(epoch_dataset, batch_size=256)\n",
        "    \n",
        "    for x, y in epoch_loader:\n",
        "        x, y = x.to(device), y.to(device)\n",
        "        optimizer.zero_grad()\n",
        "        outputs = model(x)\n",
        "        loss = criterion(outputs, y)\n",
        "        loss.backward()\n",
        "        optimizer.step()\n",
        "        \n",
        "        total_loss += loss.item() * x.size(0)\n",
        "        _, predicted = outputs.max(1)\n",
        "        total += y.size(0)\n",
        "        correct += predicted.eq(y).sum().item()\n",
        "        step += 1\n",
        "    \n",
        "    train_loss = total_loss / max(total, 1)\n",
        "    train_acc = correct / max(total, 1) * 100\n",
        "    print(f\"Epoch {epoch+1}/{NUM_EPOCHS} | Steps: {step} | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%\")\n",
        "\n",
        "print(f\"Training Complete! Total samples per epoch: {total}\")\n"
    ]

with open("train_cnn_bot.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("All critical fixes applied successfully.")
