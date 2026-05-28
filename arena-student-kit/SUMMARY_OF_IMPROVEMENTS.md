# Báo Cáo Kỹ Thuật: Tối Ưu Hóa Đối Xứng Bàn Cờ Cho Arena Bot

Tài liệu này tổng hợp chi tiết phân tích lỗi mất cân bằng tỉ lệ thắng (BLUE/RED), các công nghệ, kỹ thuật đã được áp dụng để giải quyết vấn đề và hướng dẫn triển khai.

---

## 1. Phân Tích Lỗi Mất Cân Bằng Tỉ Lệ Thắng (Root Cause Analysis)

Trước khi tối ưu hóa, Bot có tỉ lệ thắng lệch cực kỳ lớn giữa 2 bên:
- **Phe BLUE (Đi trước):** **98.40%** (246/250 trận thắng)
- **Phe RED (Đi sau):** **42.40%** (106/250 trận thắng)

Sự chênh lệch này đến từ 3 nguyên nhân cốt lõi:

### A. Lợi thế đi trước (First-Turn Advantage) của Luật Chơi
Trong game chiến thuật turn-based di chuyển và tấn công tự do, đội đi trước (BLUE) luôn có lợi thế rất lớn để áp sát, dàn trận và tấn công phủ đầu tiêu diệt sinh lực địch trước khi chúng kịp phản kháng.
*Ngay cả Heuristic Bot thuật toán chuẩn khi chạy thử nghiệm cũng bị lệch (BLUE thắng 100%, RED thắng 52% trước IntermediateBot).*

### B. Thiên lệch dữ liệu huấn luyện (Training Data Bias)
Công cụ sinh dữ liệu `DatasetGenerator.java` chỉ ghi nhận nước đi từ các trận đấu mà **Bot giành chiến thắng**. Do Bot thắng BLUE gần như tuyệt đối còn RED thua nhiều, tập dữ liệu `dataset.csv` bị áp đảo bởi các trạng thái từ góc nhìn BLUE (quân ta bên trái, quân địch bên phải). CNN bị overfit nặng nề theo hướng chơi này.

### C. Xung Đột Nhãn do Tính Chất Dịch Chuyển Bất Biến (Translation Invariance) của CNN
- Mạng CNN sử dụng 2 lớp tích chập $3 \times 3$ nên trường thụ cảm cục bộ (Receptive Field) của mỗi ô đầu ra chỉ là $5 \times 5$. Mạng không thể nhìn thấy toàn bộ bàn cờ $8 \times 8$ để biết hướng đi tuyệt đối.
- Khi chơi bên BLUE, tướng cần đi sang **phải** (tăng cột). Khi chơi bên RED, tướng cần đi sang **trái** (giảm cột).
- Do CNN có tính chất chia sẻ trọng số và thiếu thông tin tọa độ tuyệt đối, khi nhìn thấy các cấu trúc quân cục bộ giống nhau, mạng sẽ bị xung đột nhãn (phải đi trái hay phải?). Dữ liệu BLUE chiếm đa số làm CNN luôn thiên vị việc đi sang phải, khiến Bot RED thường xuyên di chuyển lỗi (đi lùi hoặc đứng im).

---

## 2. Kỹ Thuật Đối Xứng Gương ở Runtime (Horizontal Mirroring)

Để khắc phục triệt để mà không cần huấn luyện lại mô hình từ đầu, chúng ta áp dụng kỹ thuật **Horizontal Mirroring (Đối xứng gương theo trục dọc)** tại thời điểm chạy (Runtime).

### Kỹ thuật này gồm 2 bước chính:
1. **Tiền xử lý (Preprocessing - Lật input):** Nếu Bot chơi ở phe RED, toàn bộ tọa độ cột `c` của các kênh đặc trưng trong Tensor đầu vào (`active_pos`, `ally_pos`, `enemy_pos`, `dist_map`...) sẽ được lật ngược thành `7 - c`. Lúc này, mạng CNN sẽ nhìn nhận trận đấu hoàn toàn dưới góc nhìn của phe BLUE (quân ta bên trái, địch bên phải).
2. **Hậu xử lý (Postprocessing - Lật output scores):** Mạng CNN sẽ trả về ma trận điểm số (`scores`) tương ứng với góc nhìn BLUE. Trước khi đưa ra quyết định hành động thực tế trên bàn cờ, ta chỉ cần lật ngược tọa độ cột của ma trận điểm số này một lần nữa để khớp với tọa độ thực tế của phe RED.

### Ưu điểm vượt trội:
- **100% Tương thích ngược:** Không yêu cầu train lại mạng CNN hay MLP. Chỉ cần sửa đổi logic phần code Java đóng gói điều khiển.
- **Triệt tiêu xung đột nhãn:** Mạng CNN giờ đây chỉ cần học một góc nhìn duy nhất (BLUE) và chơi tốt góc nhìn đó. Khi làm RED, cơ chế mirror tự động chuyển đổi góc nhìn giúp Bot RED chơi thông minh tương đương Bot BLUE.

---

## 3. Chi Tiết Triển Khai Trong Mã Nguồn Java

### A. Hàm Helper Đối Xứng
```java
private static int getCol(int col, TeamSide side) {
    return (side == TeamSide.RED) ? (7 - col) : col;
}
```

### B. Lật Tensor Đầu Vào (playTurn)
```java
// Đánh dấu vị trí active champion (đã lật)
input[0][activeRow][getCol(activeColReal, side)] = 1.0f;

// Lật vị trí đồng minh
for (ChampionSnapshot a : view.allies()) {
    if (a.alive()) {
        int r = a.position().row(), c = a.position().col();
        int mc = getCol(c, side);
        input[1][r][mc] = 1.0f;
        input[3][r][mc] = (float) a.hp() / a.maxHp();
        input[6][r][mc] = (a.maxMana() > 0) ? (float) a.mana() / a.maxMana() : 0.0f;
        input[8][r][mc] = (float) a.attack() / 10.0f;
        input[10][r][mc] = (float) a.defense() / 10.0f;
        input[12][r][mc] = (float) a.range() / 5.0f;
    }
}
// Lật vị trí kẻ địch tương tự cho input[2], input[4], input[7], input[9], input[11], input[13]
```

### C. Lật Điểm Số Dự Đoán Đầu Ra để Lấy Hành Động
Khi chọn nước đi, ta truy vấn điểm số qua cột đối xứng:
```java
// Điểm số di chuyển (Channel 0)
float s = scores[0][nr][getCol(nc, side)];

// Điểm số tấn công (Channel 1)
float s = scores[1][enemy.position().row()][getCol(enemy.position().col(), side)];

// Điểm số dùng kỹ năng (Channel 2)
float s = scores[2][r][getCol(c, side)];
```

---

## 4. Các File Đã Được Cập Nhật Trong Hệ Thống

Chúng tôi đã cập nhật cấu trúc logic này vào các file template để đảm bảo các lần export sau tự động có tính năng đối xứng:
1. **[train_cnn_bot.py](file:///c:/Users/binhn/Downloads/arena-student-kit%204/arena-student-kit/train_cnn_bot.py):** Sửa template sinh code Java ở Bước 8.
2. **[train_cnn_bot.ipynb](file:///c:/Users/binhn/Downloads/arena-student-kit%204/arena-student-kit/train_cnn_bot.ipynb):** Đồng bộ hóa cell Jupyter Notebook xuất code Java.

### Hướng dẫn sử dụng:
- **Nếu muốn dùng luôn trọng số cũ:** Mở file `StudentBotImpl.java` (51MB) của bạn và thêm hàm `getCol`, sửa logic `playTurn` theo mẫu trên.
- **Nếu muốn train lại:** Chỉ cần chạy file Jupyter Notebook hoặc Script `train_cnn_bot.py` như bình thường, code Java xuất ra sẽ tự động chứa logic tối ưu này.
