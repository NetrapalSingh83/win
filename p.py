import time
import csv
import os
import joblib
import numpy as np
import logging
import pandas as pd
import tensorflow as tf
from logging.handlers import RotatingFileHandler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from telegram import Bot
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# === CONFIG ===
USERNAME = "8302257716"
PASSWORD = "Rajput183"
TELEGRAM_BOT_TOKEN = '6704057021:AAHPI7LcxVkUTmTZ75ulA41pU0tS0BSxm8k'
TELEGRAM_CHAT_ID = '5759284972'
LOGIN_URL = "https://2india.in/#/login"
WINGO_URL = "https://2india.in/#/home/AllLotteryGames/WinGo?id=1"

CSV_FILE = "wingo_data.csv"
MODEL_LSTM = "lstm_model.h5"
MODEL_MLP = "mlp_model.pkl"
WINDOW_SIZE = 5
NUM_CLASSES = 10

# === LOGGER SETUP ===
logger = logging.getLogger("WingoUltimateBot")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler = RotatingFileHandler("wingo_bot.log", maxBytes=1000000, backupCount=3, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# === UTILITY FUNCTIONS ===
def save_result(period, number):
    write_header = not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(['period', 'number'])
        writer.writerow([period, number])

def save_prediction(period, prediction, next_period):
    write_header = not os.path.exists("prediction_log.csv") or os.path.getsize("prediction_log.csv") == 0
    with open("prediction_log.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(['predicted_for_period', 'predicted_number', 'next_period'])
        writer.writerow([period, prediction, next_period])

def load_all_numbers():
    numbers = []
    rows = []
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                rows.append(row)
        rows.sort(key=lambda x: int(x["period"]))
        for row in rows:
            try:
                num = int(row["number"])
                numbers.append(num)
            except:
                continue
    return numbers

def get_logged_periods():
    periods = set()
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                periods.add(row['period'])
    return periods

def collect_page_data(driver):
    data = []
    rows = driver.find_elements(By.CSS_SELECTOR, '.GameRecord__C-body .van-row')
    for row in rows:
        try:
            cols = row.find_elements(By.CLASS_NAME, 'van-col')
            if len(cols) >= 2:
                period = cols[0].text.strip()
                number_el = cols[1].find_element(By.CLASS_NAME, 'GameRecord__C-body-num')
                number = int(number_el.text.strip())
                data.append((period, number))
        except:
            continue
    return data

def login(driver):
    logger.info("Logging in...")
    driver.get(LOGIN_URL)
    time.sleep(2)
    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, 'input[name="userNumber"]'))
    ).send_keys(USERNAME)
    driver.find_element(By.CSS_SELECTOR, 'input[type="password"]').send_keys(PASSWORD)
    driver.find_element(By.CSS_SELECTOR, 'button.active').click()
    time.sleep(5)

# === ML FUNCTIONS ===
def train_lstm_model():
    if not os.path.exists(CSV_FILE):
        return
    df = pd.read_csv(CSV_FILE)
    df = df.sort_values(by='period')
    numbers = df['number'].astype(int).tolist()
    if len(numbers) < 20:
        logger.warning("Not enough data for LSTM training.")
        return
    X, y = [], []
    for i in range(len(numbers) - WINDOW_SIZE):
        X.append(numbers[i:i+WINDOW_SIZE])
        y.append(numbers[i+WINDOW_SIZE])

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = tf.keras.utils.to_categorical(y, num_classes=NUM_CLASSES)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = Sequential()
    model.add(LSTM(64, input_shape=(WINDOW_SIZE, 1)))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)
    model.save(MODEL_LSTM)
    logger.info("✅ LSTM model trained and saved.")

def train_mlp_model():
    if not os.path.exists(CSV_FILE):
        return
    df = pd.read_csv(CSV_FILE)
    df = df.sort_values(by='period')
    numbers = df['number'].astype(int).tolist()
    if len(numbers) < 20:
        logger.warning("Not enough data for MLP training.")
        return
    X, y = [], []
    for i in range(len(numbers) - WINDOW_SIZE):
        X.append(numbers[i:i+WINDOW_SIZE])
        y.append(numbers[i+WINDOW_SIZE])

    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
    model.fit(X, y)
    joblib.dump(model, MODEL_MLP)
    logger.info("✅ MLP model trained and saved.")

def predict_next_number(numbers):
    if len(numbers) < WINDOW_SIZE:
        return 0

    try:
        if os.path.exists(MODEL_LSTM):
            model = load_model(MODEL_LSTM)
            input_seq = np.array(numbers[-WINDOW_SIZE:]).reshape((1, WINDOW_SIZE, 1))
            pred = model.predict(input_seq)
            return np.argmax(pred, axis=1)[0]
    except Exception as e:
        logger.warning("⚠️ LSTM failed. Trying MLP...", exc_info=True)

    try:
        if os.path.exists(MODEL_MLP):
            model = joblib.load(MODEL_MLP)
            input_data = np.array(numbers[-WINDOW_SIZE:]).reshape(1, -1)
            pred = model.predict(input_data)[0]
            return int(pred)
    except Exception as e:
        logger.error("❌ MLP fallback failed too.", exc_info=True)
    
    return 0

# === CLASSIFICATION
def get_size_label(number):
    return "🔺 *Big* 🟢" if number >= 5 else "🔻 *Small* 🔴"

def get_color_label(number):
    if number == 0:
        return "🟣 Purple"
    elif number in [1, 3, 5, 7, 9]:
        return "🟢 Green"
    elif number in [2, 4, 6, 8]:
        return "🔴 Red"
    else:
        return "❓ Unknown"

def send_to_telegram(prediction, next_period):
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    size_label = get_size_label(prediction)
    color_label = get_color_label(prediction)
    msg = (
        "🎯 *Wingo Ultimate Bot*\n"
        f"📈 Next Period: `{next_period}`\n"
        f"🔮 Predicted Number: *{prediction}*\n"
        f"{size_label} | {color_label}"
    )
    bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode='Markdown')

# === MONITOR
def monitor_results(driver, known_periods):
    logger.info("Monitoring Wingo...")
    while True:
        try:
            driver.get(WINGO_URL)
            time.sleep(5)
            data = collect_page_data(driver)

            for period, number in data:
                if period not in known_periods:
                    logger.info(f"New Result: {period} → {number}")
                    known_periods.add(period)
                    save_result(period, number)

                    train_lstm_model()
                    train_mlp_model()

                    all_numbers = load_all_numbers()
                    prediction = predict_next_number(all_numbers)

                    next_period = str(int(period) + 1)
                    logger.info(f"Prediction: {prediction} for next period: {next_period}")
                    send_to_telegram(prediction, next_period)
                    save_prediction(period, prediction, next_period)
            time.sleep(10)
        except Exception as e:
            logger.error("Monitor loop error", exc_info=True)
            login(driver)

# === MAIN
def main():
    logger.info("🚀 Launching Wingo Ultimate Bot...")
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # Uncomment on VPS
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(options=chrome_options)

    try:
        login(driver)
        driver.get(WINGO_URL)
        time.sleep(5)
        known_periods = get_logged_periods()
        monitor_results(driver, known_periods)
    except Exception as e:
        logger.error("Fatal error in main()", exc_info=True)
    finally:
        logger.info("Shutting down bot.")

if __name__ == "__main__":
    main()
