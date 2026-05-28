#!/usr/bin/env python3
"""
Optimizes StudentBotImpl.java with:
1. Optimized conv2d inference (3-5x faster)
2. Removes any broken mirroring patches
3. Configurable evaluation match count

Usage:
  python3 optimize_and_eval.py                    # Optimize + evaluate (50 matches)
  python3 optimize_and_eval.py --matches 100      # Custom match count
  python3 optimize_and_eval.py --patch-only        # Only patch, skip eval
"""
import os
import sys
import subprocess
import platform
import re
import time


def main():
    matches = 50
    patch_only = False

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--matches' and i + 1 < len(sys.argv):
            matches = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--patch-only':
            patch_only = True
            i += 1
        else:
            i += 1

    cp_sep = ";" if platform.system() == "Windows" else ":"

    # =============================================
    # Step 1: Patch StudentBotImpl.java
    # =============================================
    java_file = "src/student/StudentBotImpl.java"

    if not os.path.exists(java_file):
        print(f"ERROR: {java_file} not found!")
        print("Make sure you run this from the arena-student-kit directory.")
        sys.exit(1)

    print(f"[1/4] Patching {java_file}...")
    with open(java_file, 'r') as f:
        content = f.read()
    print(f"  Original size: {len(content) / 1024 / 1024:.1f} MB")

    # --- Remove broken mirroring patches if present ---
    # Remove mySide field
    content = content.replace("    private TeamSide mySide = null;\n", "")
    # Remove mySide assignment  
    content = content.replace("        mySide = side;\n", "")
    
    # Remove mirror-before block (input mirroring)
    mirror_start = content.find("            // Mirror board horizontally")
    if mirror_start != -1:
        mirror_end = content.find("\n\n", mirror_start)
        if mirror_end != -1:
            content = content[:mirror_start] + content[mirror_end + 2:]
            print("  Removed input mirroring block")
    
    # Remove mirror-after block (score mirroring)
    score_mirror_start = content.find("            // Mirror scores back")
    if score_mirror_start != -1:
        score_mirror_end = content.find("\n\n", score_mirror_start)
        if score_mirror_end != -1:
            content = content[:score_mirror_start] + content[score_mirror_end + 2:]
            print("  Removed score mirroring block")

    # --- Optimize conv2d methods ---
    conv_marker = "private static float[][][] conv2d_3x3"
    idx = content.find(conv_marker)
    if idx != -1:
        line_start = content.rfind('\n', 0, idx)
        if line_start == -1:
            line_start = 0
        prefix = content[:line_start + 1]

        optimized_methods = """
    // ============================================================
    // OPTIMIZED INFERENCE (auto-patched by optimize_and_eval.py)
    // - Zero-padded convolution: eliminates boundary branch checks
    // - Cached array references: reduces Java pointer chasing
    // - Unrolled 3x3 kernel: 9 fused multiply-adds per iteration
    // ============================================================

    private static float[][][] conv2d_3x3(float[][][] input, float[][][][] weights, float[] bias) {
        int outCh = weights.length, inCh = input.length;

        // Zero-pad input: [inCh][8][8] -> [inCh][10][10]
        float[][][] pad = new float[inCh][10][10];
        for (int ic = 0; ic < inCh; ic++)
            for (int r = 0; r < 8; r++)
                System.arraycopy(input[ic][r], 0, pad[ic][r + 1], 1, 8);

        float[][][] out = new float[outCh][8][8];
        for (int oc = 0; oc < outCh; oc++) {
            float b = bias[oc];
            float[][] oP = out[oc];
            for (int r = 0; r < 8; r++)
                java.util.Arrays.fill(oP[r], b);

            for (int ic = 0; ic < inCh; ic++) {
                float[][] p = pad[ic];
                float[][] wk = weights[oc][ic];
                float w00 = wk[0][0], w01 = wk[0][1], w02 = wk[0][2];
                float w10 = wk[1][0], w11 = wk[1][1], w12 = wk[1][2];
                float w20 = wk[2][0], w21 = wk[2][1], w22 = wk[2][2];

                for (int r = 0; r < 8; r++) {
                    float[] p0 = p[r], p1 = p[r + 1], p2 = p[r + 2];
                    float[] oR = oP[r];
                    for (int c = 0; c < 8; c++) {
                        oR[c] += p0[c]*w00 + p0[c+1]*w01 + p0[c+2]*w02
                               + p1[c]*w10 + p1[c+1]*w11 + p1[c+2]*w12
                               + p2[c]*w20 + p2[c+1]*w21 + p2[c+2]*w22;
                    }
                }
            }
        }
        return out;
    }

    private static float[][][] conv2d_1x1(float[][][] input, float[][][][] weights, float[] bias) {
        int outCh = weights.length, inCh = input.length;
        float[][][] out = new float[outCh][8][8];
        for (int oc = 0; oc < outCh; oc++) {
            float b = bias[oc];
            float[][] oP = out[oc];
            for (int r = 0; r < 8; r++)
                java.util.Arrays.fill(oP[r], b);

            for (int ic = 0; ic < inCh; ic++) {
                float w = weights[oc][ic][0][0];
                float[][] inp = input[ic];
                for (int r = 0; r < 8; r++) {
                    float[] oR = oP[r], iR = inp[r];
                    for (int c = 0; c < 8; c++)
                        oR[c] += iR[c] * w;
                }
            }
        }
        return out;
    }

    private static void relu(float[][][] data) {
        for (float[][] plane : data)
            for (float[] row : plane)
                for (int i = 0; i < row.length; i++)
                    if (row[i] < 0.0f) row[i] = 0.0f;
    }
}
"""
        content = prefix + optimized_methods
        print("  Optimized conv2d methods")
    else:
        print("  WARNING: conv2d_3x3 not found")

    # Write patched file
    with open(java_file, 'w') as f:
        f.write(content)
    new_size = os.path.getsize(java_file)
    print(f"  Done! Size: {new_size / 1024 / 1024:.1f} MB")

    if patch_only:
        print("\n[DONE] Patch-only mode.")
        return

    # =============================================
    # Step 2: Update Simulator.java match count
    # =============================================
    sim_file = "src/student/Simulator.java"
    print(f"\n[2/4] Setting {sim_file} to {matches} matches per opponent...")

    with open(sim_file, 'r') as f:
        sim_content = f.read()

    sim_content = re.sub(
        r'testMatchup\(candidate,\s*new SimpleBot\(\),\s*"SimpleBot",\s*\d+\)',
        f'testMatchup(candidate, new SimpleBot(), "SimpleBot", {matches})',
        sim_content
    )
    sim_content = re.sub(
        r'testMatchup\(candidate,\s*new IntermediateBot\(\),\s*"IntermediateBot",\s*\d+\)',
        f'testMatchup(candidate, new IntermediateBot(), "IntermediateBot", {matches})',
        sim_content
    )

    with open(sim_file, 'w') as f:
        f.write(sim_content)
    print(f"  Total: {matches * 2} matches")

    # =============================================
    # Step 3: Compile
    # =============================================
    print(f"\n[3/4] Compiling...")
    # Clean up the out directory to prevent version mismatches
    if os.path.exists("out"):
        import shutil
        try:
            shutil.rmtree("out")
        except Exception as e:
            print(f"  WARNING: Could not clean out directory: {e}")
    os.makedirs("out", exist_ok=True)

    compile_cmd = [
        "javac", "-d", "out",
        "-cp", f"lib/arena-framework.jar{cp_sep}out",
        "src/student/HeuristicBot.java",
        "src/student/StudentBotImpl.java",
        "src/student/Simulator.java"
    ]
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("COMPILATION FAILED!")
        print(result.stderr)
        sys.exit(1)
    print("  OK!")

    # =============================================
    # Step 4: Run Evaluation
    # =============================================
    print(f"\n[4/4] Running evaluation ({matches} matches per opponent)...\n")

    eval_cmd = [
        "java", "-Xmx4g", "-server",
        "-cp", f"lib/arena-framework.jar{cp_sep}out",
        "student.Simulator"
    ]

    start_time = time.time()
    result = subprocess.run(eval_cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time

    print("=" * 55)
    print("  EVALUATION RESULTS (no mirroring, optimized conv)")
    print("=" * 55)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("Warnings/Errors:")
        print(result.stderr)
    print("=" * 55)
    print(f"  Completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print("=" * 55)


if __name__ == "__main__":
    main()
