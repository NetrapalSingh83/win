import time
import csv
import os
import joblib
import numpy as np
import logging
import pandas as pd
from logging.handlers import RotatingFileHandler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from telegram import Bot

# === CONFIG ===
USERNAME = "8302257716"
PASSWORD = "Rajput183"
TELEGRAM_BOT_TOKEN = '7694295301:AAFq9WhPnAYFdr_ZOMHISAAkP8ZGaDit-Nw'
TELEGRAM_CHAT_IDS = ['5759284972', '5851079012']  # Add multiple chat IDs here

LOGIN_URL = "https://2india.in/#/login"
WINGO_URL = "https://2india.in/#/home/AllLotteryGames/WinGo?id=1"

CSV_FILE = "wingo_data.csv"
LOG_FILE = "prediction_log.csv"
MODEL_FILE = "ml_model.pkl"
WINDOW_SIZE = 5

# === LOGGER SETUP ===
logger = logging.getLogger("WingoBot")
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
    write_header = not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0
    with open(LOG_FILE, mode='a', newline='') as file:
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
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'input[name="userNumber"]'))
        ).send_keys(USERNAME)
        driver.find_element(By.CSS_SELECTOR, 'input[type="password"]').send_keys(PASSWORD)
        driver.find_element(By.CSS_SELECTOR, 'button.active').click()
        time.sleep(5)
    except Exception as e:
        logger.error("Login failed", exc_info=True)

# === ML CORE ===
def train_model():
    if not os.path.exists(CSV_FILE):
        logger.warning("No data to train model.")
        return

    df = pd.read_csv(CSV_FILE)
    df = df.sort_values(by='period')
    numbers = df['number'].astype(int).tolist()
    X, y = [], []

    for i in range(len(numbers) - WINDOW_SIZE):
        X.append(numbers[i:i+WINDOW_SIZE])
        y.append(numbers[i+WINDOW_SIZE])

    if len(X) < 10:
        logger.warning("Insufficient data to train ML model.")
        return

    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_FILE)
    acc = model.score(X_test, y_test)
    logger.info(f"ML model trained. Accuracy: {acc:.2%}")

def predict_next_number(numbers):
    if len(numbers) < WINDOW_SIZE:
        logger.warning("Not enough numbers for prediction.")
        return 0

    if not os.path.exists(MODEL_FILE):
        logger.info("Model not found. Training now...")
        train_model()
    if not os.path.exists(MODEL_FILE):
        logger.error("ML model still missing after train attempt.")
        return 0

    try:
        model = joblib.load(MODEL_FILE)
        input_data = np.array(numbers[-WINDOW_SIZE:]).reshape(1, -1)
        prediction = model.predict(input_data)[0]
        return int(prediction)
    except Exception as e:
        logger.error("Prediction failed.", exc_info=True)
        return 0

# === CLASSIFICATION LABELS ===
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

# === TELEGRAM NOTIFY ===
def send_to_telegram(prediction, next_period):
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    size_label = get_size_label(prediction)
    color_label = get_color_label(prediction)
    msg = (
        "🎯 *Wingo ML Bot*\n"
        f"📈 Next Period: `{next_period}`\n"
        f"🔮 Predicted Number: *{prediction}*\n"
        f"{size_label} | {color_label}"
    )
    for chat_id in TELEGRAM_CHAT_IDS:
        try:
            bot.send_message(chat_id=chat_id, text=msg, parse_mode='Markdown')
            logger.info(f"✅ Sent prediction to {chat_id}")
        except Exception as e:
            logger.error(f"❌ Failed to send to {chat_id}", exc_info=True)

# === MONITOR LOOP WITH AUTO-RELOAD ===
def monitor_results(driver, known_periods):
    logger.info("📡 Monitoring Wingo...")

    last_refresh_time = time.time()

    while True:
        try:
            current_time = time.time()

            # 🔁 Refresh page every 30 seconds
            if current_time - last_refresh_time >= 30:
                logger.info("🔄 Auto-refreshing Wingo page (30s interval)")
                driver.refresh()
                last_refresh_time = current_time
                time.sleep(5)  # Wait a bit for page to reload

            data = collect_page_data(driver)

            if not data:
                logger.warning("⚠️ No data found on page.")
                time.sleep(5)
                continue

            for period, number in data:
                if period not in known_periods:
                    logger.info(f"🆕 New Result: {period} → {number}")
                    known_periods.add(period)
                    save_result(period, number)

                    train_model()
                    all_numbers = load_all_numbers()
                    prediction = predict_next_number(all_numbers)

                    next_period = str(int(period) + 1)
                    logger.info(f"🔮 Prediction: {prediction} for next period: {next_period}")
                    send_to_telegram(prediction, next_period)
                    save_prediction(period, prediction, next_period)

            time.sleep(10)

        except Exception as e:
            logger.error("❌ Error during monitor loop. Retrying...", exc_info=True)
            time.sleep(5)

# === MAIN ENTRY ===
def main():
    logger.info("🚀 Starting Wingo ML Bot...")
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # Uncomment for headless mode
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
        logger.error("Fatal error in main", exc_info=True)
    finally:
        logger.info("Bot shutting down.")

if __name__ == "__main__":
    main()
