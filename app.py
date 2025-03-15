from flask import Flask, render_template, request
import math
import warnings
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from keras.models import load_model
from data.data import process_data
from flask import Flask, render_template
from flask import redirect, url_for, request, flash, session
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, SelectField
from wtforms.validators import DataRequired, Length, EqualTo, ValidationError
from passlib.context import CryptContext
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from functools import wraps
import pymysql

import pandas as pd
import os
from os.path import join as opj
import re
import time
import datetime
import json
import numpy as np
import geatpy as ea
from utils import residual_square
import utils
import math

app = Flask(__name__)


app.jinja_env.globals.update(zip=zip)

# 关闭警告
warnings.filterwarnings("ignore")

app.secret_key = "your-secret-key"  # Required for session management and CSRF protection

# 配置数据库连接
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://traffic_user:traffic_password@localhost:3306/traffic_flow'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 初始化数据库
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 登录验证装饰器
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('请先登录', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# 用户模型
class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    hashed_password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='user')
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    def __repr__(self):
        return f'<User {self.username}>'
    
    def verify_password(self, password):
        return pwd_context.verify(password, self.hashed_password)

# 创建数据库表
@app.before_first_request
def create_tables():
    db.create_all()
    # 检查是否有管理员用户，如果没有则创建一个
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        admin = User(
            username='admin',
            hashed_password=pwd_context.hash('admin123'),
            role='admin'
        )
        db.session.add(admin)
        db.session.commit()

# Login form
class LoginForm(FlaskForm):
    username = StringField("用户名", validators=[DataRequired()])
    password = PasswordField("密码", validators=[DataRequired()])
    submit = SubmitField("登录")

# Registration form
class RegistrationForm(FlaskForm):
    username = StringField("用户名", validators=[DataRequired(), Length(min=3, max=50)])
    password = PasswordField("密码", validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField("确认密码", validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField("注册")
    
    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('该用户名已被使用，请选择其他用户名')

# User management form
class UserForm(FlaskForm):
    username = StringField("用户名", validators=[DataRequired()])
    password = PasswordField("密码", validators=[DataRequired()])
    confirm_password = PasswordField("确认密码", validators=[DataRequired(), EqualTo('password')])
    role = SelectField("角色", choices=[("user", "用户"), ("admin", "管理员")], validators=[DataRequired()])
    submit = SubmitField("提交")

# Function to authenticate user
def authenticate_user(username, password):
    user = User.query.filter_by(username=username).first()
    if not user or not user.verify_password(password):
        return False
    return user

def MAPE(y_true, y_pred):
    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    num = len(y_pred)
    sums = 0

    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp

    mape = sums * (100 / num)

    return mape

def eva_regress(y_true, y_pred):
    mape = MAPE(y_true, y_pred)
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    return {
        'explained_variance_score': vs,
        'mape': mape,
        'mae': mae,
        'mse': mse,
        'rmse': math.sqrt(mse),
        'r2': r2
    }

@app.route("/login", methods=["GET", "POST"])
def login():
    if "username" in session:
        return redirect(url_for("index"))

    form = LoginForm()
    register_form = RegistrationForm()
    
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        user = authenticate_user(username, password)
        if user:
            session["username"] = username  # Store username in session
            session["role"] = user.role  # Store user role in session
            return redirect(url_for("index"))
        else:
            flash("用户名或密码不正确", "danger")
    
    return render_template("login.html", form=form, register_form=register_form)

@app.route("/register", methods=["GET", "POST"])
def register():
    register_form = RegistrationForm()
    if register_form.validate_on_submit():
        hashed_password = pwd_context.hash(register_form.password.data)
        new_user = User(
            username=register_form.username.data,
            hashed_password=hashed_password,
            role="user"  # 默认为普通用户角色
        )
        db.session.add(new_user)
        db.session.commit()
        flash("注册成功！现在您可以登录了。", "success")
        return redirect(url_for("login"))
    
    # 如果表单验证失败，返回登录页面并显示错误
    form = LoginForm()
    return render_template("register.html", form=register_form)

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    params = {
        'models': ['LSTM', 'GRU', 'SAEs'],  # 所有模型
        'lag': 12,
        'train_file': 'data/100211data/100211_weekend_train.csv',
        'test_file': 'data/100211data/100211_weekend_test.csv'
    }

    if request.method == 'POST':
        try:
            # 加载所有模型
            models = {}
            for model_name in params['models']:
                try:
                    model_path = f"model/100211_all/{model_name.lower()}.h5"
                    models[model_name] = load_model(model_path)
                    print(f"成功加载模型: {model_name} 从 {model_path}")
                except Exception as e:
                    print(f"加载模型 {model_name} 时出错: {str(e)}")
                    flash(f"加载模型 {model_name} 时出错: {str(e)}", "danger")
                    return render_template('index.html', params=params, error=f"加载模型失败: {str(e)}")

            # 处理数据
            try:
                X_train, y_train, X_test, y_test, scaler = process_data(params['train_file'], params['test_file'], params['lag'])
                print(f"数据处理完成:")
                print(f"X_train shape: {X_train.shape}")
                print(f"y_train shape: {y_train.shape}")
                print(f"X_test shape: {X_test.shape}")
                print(f"y_test shape: {y_test.shape}")
                
                # 转换y_test为原始值
                y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]
                print(f"y_test 转换后 shape: {y_test.shape}")
                print(f"y_test 类型: {type(y_test)}")
                print(f"y_test 前5个值: {y_test[:5]}")
            except Exception as e:
                print(f"处理数据时出错: {str(e)}")
                flash(f"处理数据时出错: {str(e)}", "danger")
                return render_template('index.html', params=params, error=f"数据处理失败: {str(e)}")

            # 存储所有模型的预测结果和评估结果
            y_preds = []
            evaluation_results = []
            
            for name in params['models']:
                try:
                    model = models[name]
                    # 复制测试数据以避免修改原始数据
                    X_test_copy = X_test.copy()
                    print(f"处理模型: {name}")
                    
                    # 根据模型类型重塑输入数据
                    if name == 'SAEs':
                        X_test_reshaped = np.reshape(X_test_copy, (X_test_copy.shape[0], X_test_copy.shape[1]))
                        print(f"SAEs X_test_reshaped shape: {X_test_reshaped.shape}")
                    else:
                        X_test_reshaped = np.reshape(X_test_copy, (X_test_copy.shape[0], X_test_copy.shape[1], 1))
                        print(f"{name} X_test_reshaped shape: {X_test_reshaped.shape}")
                    
                    # 进行预测
                    predicted = model.predict(X_test_reshaped)
                    print(f"原始预测 shape: {predicted.shape}")
                    
                    # 将预测值转换回原始范围
                    predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
                    print(f"预测值转换后 shape: {predicted.shape}")
                    print(f"预测值类型: {type(predicted)}")
                    print(f"预测值前5个: {predicted[:5]}")
                    
                    # 确保预测数据是有效的数字
                    predicted_cleaned = np.array([float(x) if not np.isnan(float(x)) else 0.0 for x in predicted])
                    
                    # 计算评估指标
                    eval_result = eva_regress(y_test, predicted_cleaned)
                    evaluation_results.append(eval_result)
                    print(f"模型 {name} 评估结果: {eval_result}")
                    
                    # 转换为列表并确保长度为288
                    pred_list = predicted_cleaned.tolist()
                    print(f"pred_list 类型: {type(pred_list)}")
                    print(f"调整前 pred_list 长度: {len(pred_list)}")
                    
                    # 调整预测列表长度为288
                    if len(pred_list) > 288:
                        pred_list = pred_list[:288]
                    elif len(pred_list) < 288:
                        # 如果预测数据少于288，用最后一个值填充
                        last_value = pred_list[-1] if pred_list else 0.0
                        pred_list.extend([last_value] * (288 - len(pred_list)))
                    
                    print(f"调整后 pred_list 长度: {len(pred_list)}")
                    print(f"pred_list 前5个值: {pred_list[:5]}")
                    
                    # 添加到预测结果列表
                    y_preds.append(pred_list)
                except Exception as e:
                    print(f"处理模型 {name} 时出错: {str(e)}")
                    # 如果某个模型出错，添加空列表作为占位符
                    y_preds.append([0.0] * 288)
                    evaluation_results.append({
                        'explained_variance_score': 0,
                        'mape': 0,
                        'mae': 0,
                        'mse': 0,
                        'rmse': 0,
                        'r2': 0,
                        'error': str(e)
                    })

            # 确保y_test也是正确的长度和格式
            try:
                # 转换为Python列表
                y_test_list = y_test.tolist()
                print(f"y_test_list 类型: {type(y_test_list)}")
                print(f"调整前 y_test_list 长度: {len(y_test_list)}")
                
                # 确保y_test_list中没有NaN值
                y_test_list = [float(x) if not math.isnan(float(x)) else 0.0 for x in y_test_list]
                
                # 调整y_test_list长度为288
                if len(y_test_list) > 288:
                    y_test_list = y_test_list[:288]
                elif len(y_test_list) < 288:
                    last_value = y_test_list[-1] if y_test_list else 0.0
                    y_test_list.extend([last_value] * (288 - len(y_test_list)))
                
                print(f"调整后 y_test_list 长度: {len(y_test_list)}")
                print(f"y_test_list 前5个值: {y_test_list[:5]}")
            except Exception as e:
                print(f"处理y_test_list时出错: {str(e)}")
                y_test_list = [0.0] * 288

            print(f"y_preds 类型: {type(y_preds)}")
            print(f"y_preds 长度: {len(y_preds)}")
            for i, pred in enumerate(y_preds):
                print(f"y_preds[{i}] 类型: {type(pred)}")
                print(f"y_preds[{i}] 长度: {len(pred)}")
                print(f"y_preds[{i}] 前5个值: {pred[:5]}")

            # 返回所有数据给前端
            return render_template('index.html',
                                y_true=y_test_list,
                                y_preds=y_preds,
                                evaluation_results=evaluation_results,
                                params=params)
        except Exception as e:
            print(f"处理请求时出现错误: {str(e)}")
            flash(f"处理请求时出现错误: {str(e)}", "danger")
            return render_template('index.html', params=params, error=f"处理请求失败: {str(e)}")
    else:
        return render_template('index.html', params=params)

@app.route("/logout")
@login_required
def logout():
    session.pop("username", None)
    session.pop("role", None)
    return redirect(url_for("login"))


# User management routes
@app.route("/manage_users")
@login_required
def manage_users():
    if session.get('role') != 'admin':
        flash('您没有管理员权限', 'danger')
        return redirect(url_for('index'))
    
    users = User.query.all()
    return render_template('manage_users.html', users=users)

@app.route("/delete_user/<username>", methods=["POST"])
@login_required
def delete_user(username):
    if session.get("role") != "admin":
        flash("您没有管理员权限", "danger")
        return redirect(url_for("index"))
    
    # 不允许删除当前登录的用户
    if username == session["username"]:
        flash("不能删除当前登录的用户", "danger")
        return redirect(url_for("manage_users"))
    
    user = User.query.filter_by(username=username).first()
    if user:
        db.session.delete(user)
        db.session.commit()
        flash(f"用户 {username} 已成功删除", "success")
    
    return redirect(url_for("manage_users"))

@app.route("/edit_user/<username>", methods=["GET", "POST"])
@login_required
def edit_user(username):
    if session.get("role") != "admin":
        flash("您没有管理员权限", "danger")
        return redirect(url_for("index"))
    
    user = User.query.filter_by(username=username).first()
    if not user:
        flash(f"用户 {username} 不存在", "danger")
        return redirect(url_for("manage_users"))
    
    form = UserForm(obj=user)
    # 在GET请求时清空密码字段
    if request.method == "GET":
        form.password.data = ""
        form.confirm_password.data = ""
        
    if form.validate_on_submit():
        # 检查用户名是否已存在（如果用户名已更改）
        if user.username != form.username.data:
            existing_user = User.query.filter_by(username=form.username.data).first()
            if existing_user:
                flash("用户名已存在", "danger")
                return render_template("edit_user.html", form=form, username=username)
        
        # 更新用户信息
        user.username = form.username.data
        if form.password.data:  # 只有当提供了密码时才更新密码
            user.hashed_password = pwd_context.hash(form.password.data)
        user.role = form.role.data
        db.session.commit()
        flash(f"用户 {username} 已成功更新", "success")
        return redirect(url_for("manage_users"))
    
    return render_template("edit_user.html", form=form, username=username)

@app.route('/create_user', methods=['GET', 'POST'])
@login_required
def create_user():
    # 检查当前用户是否为管理员
    if session.get('role') != 'admin':
        flash('您没有管理员权限', 'danger')
        return redirect(url_for('index'))
    
    form = UserForm()
    if form.validate_on_submit():
        # 检查用户名是否已存在
        existing_user = User.query.filter_by(username=form.username.data).first()
        if existing_user:
            flash('用户名已存在', 'danger')
            return render_template('create_user.html', form=form)
        
        # 创建新用户
        new_user = User(
            username=form.username.data,
            hashed_password=pwd_context.hash(form.password.data),
            role=form.role.data
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('用户创建成功', 'success')
        return redirect(url_for('manage_users'))
    
    return render_template('create_user.html', form=form)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
