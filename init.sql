CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 添加默认管理员用户 (密码: admin123)
INSERT INTO users (username, hashed_password, role) VALUES 
('admin', '$2b$12$dD0crDldydRyxClx9JNdT.iL4z4DQX2izKgcaci6hmVl37T1fr2Qa', 'admin');
