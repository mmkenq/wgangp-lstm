# === GAN & other
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import tempfile
import hashlib
import cv2
import time
import numpy as np
import io

# === LSTM ===
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import Counter

# === TG BOT ===
import telebot
from telebot import types
from telebot.types import InputMediaPhoto

# === FOR Radeon 680M ONLY ===
# comment it if u have another gpu
#from os import putenv
#putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

# Константы
BASE_DIR = "./datasets"
GENERATED_IMAGES_DIR = "generated_images"
LOG_EPOCH_NUM=50
G_LEARNING_RATE=0.0008
D_LEARNING_RATE=0.0002
SEGMENGT_ISOLATE_THRESHOLD=0.3

# Параметры
img_size = 128
batch_size = 64
latent_dim = 100
bot = telebot.TeleBot('1222251184:AAFnzETQIRRTV_0tY-GrLOJartao94-nHjY')

# Проверка доступности CUDA
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Используется GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Используется CPU")

# Проверка, на каком устройстве находится тензор
x = torch.tensor([1, 2, 3]).to(device)
print(f"Тензор находится на устройстве: {x.device}")

# Путь для сохранения изображений
os.makedirs(f"{GENERATED_IMAGES_DIR}", exist_ok=True)


def get_subset_hash(subset_path):
    files = sorted([f for f in os.listdir(subset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    hash_md5 = hashlib.md5()
    for fname in files:
        hash_md5.update(fname.encode())
    return hash_md5.hexdigest()

def save_models(category, generator, discriminator):
    os.makedirs('models', exist_ok=True)
    torch.save(generator.state_dict(), f'models/{category}_generator.pth')
    torch.save(discriminator.state_dict(), f'models/{category}_discriminator.pth')

def load_models(category, generator, discriminator):
    g_path = f'models/{category}_generator.pth'
    d_path = f'models/{category}_discriminator.pth'
    if os.path.exists(g_path) and os.path.exists(d_path):
        generator.load_state_dict(torch.load(g_path, map_location=device))
        discriminator.load_state_dict(torch.load(d_path, map_location=device))
        return True
    return False

def train_or_load(category, subset_path, generator, discriminator, epochs, message_chat_id):
    hash_file = f"models/{category}_subset_hash.json"
    current_hash = get_subset_hash(subset_path)
    need_train = True

    # Проверяем, есть ли сохранённый хэш
    if os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            data = json.load(f)
        if data.get('hash') == current_hash:
            # Если хэш совпадает, пробуем загрузить модель
            if load_models(category, generator, discriminator):
                print(f"Модель для {category} загружена из кэша.")
                need_train = False
                save_generated_images(epoch=0, message_chat_id=message_chat_id, epochs=1, batch_idx=0)
                bot.send_message(message_chat_id, "(Модель была загружена из кэша)")
            else:
                print(f"Нет сохранённой модели для {category}, будет обучение.")
        else:
            print(f"В subset {category} появились новые изображения, будет переобучение.")
    else:
        print(f"Нет информации о subset {category}, будет обучение.")

    if need_train:
        # Отправляем на обучение
        # После обучения сохраняем модель и новый хэш
        relearn(message_chat_id, category, epochs)
        save_models(category, generator, discriminator)
        with open(hash_file, 'w') as f:
            json.dump({'hash': current_hash}, f)


# === LSTM START ===
def tokenize(text):
    return text.lower().split()

def build_vocab(texts, max_size=10000):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {'<pad>': 0, '<unk>': 1}
    for word, _ in counter.most_common(max_size-2):
        vocab[word] = len(vocab)
    return vocab

def text_to_sequence(text, vocab, max_length=20):
    tokens = tokenize(text)
    sequence = [vocab.get(t, 1) for t in tokens][:max_length]
    if len(sequence) < max_length:
        sequence += [0]*(max_length - len(sequence))
    return sequence

def train_lstm_classifier(dataset, vocab_size, num_classes, device, epochs=10):
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    model = TextLSTMClassifier(vocab_size, 128, 128, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total, correct = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        print(f'LSTM Epoch {epoch+1}: acc={correct/total:.2f}')
    return model

class TextCategoryDataset(Dataset):
    def __init__(self, base_dir, vocab, class_to_idx, max_length=20):
        self.samples = []
        self.vocab = vocab
        self.class_to_idx = class_to_idx
        self.max_length = max_length
        for cat in os.listdir(base_dir):
            cat_path = os.path.join(base_dir, cat)
            if not os.path.isdir(cat_path): continue
            for file in os.listdir(cat_path):
                if file.endswith('.txt'):
                    with open(os.path.join(cat_path, file), 'r') as f:
                        text = f.read().strip()
                    self.samples.append((text, self.class_to_idx[cat]))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        text, label = self.samples[idx]
        seq = torch.tensor(text_to_sequence(text, self.vocab, self.max_length), dtype=torch.long)
        return seq, label

class TextLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        emb = self.embedding(x)
        _, (h, _) = self.lstm(emb)
        out = self.fc(h[-1])
        return out
# === LSTM END ===


# === GAN START ===
# Трансформации и загрузка датасета
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size(), device=device, requires_grad=False)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

class OptimizedResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Depthwise convolution
        self.depthwise = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                     padding=1, groups=in_channels))
        # Pointwise convolution
        self.pointwise = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, in_channels, 1))
        self.norm = nn.InstanceNorm2d(in_channels, affine=True)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x += residual
        return self.activation(x)

# Generator
class EnhancedGenerator(nn.Module):
    def __init__(self, latent_dim=100, channels=3):
        super().__init__()
        
        # Начальный проекционный слой
        self.proj = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Optimized Residual-блоки
        # (4 оптимизированных блока)
        self.res_blocks = nn.Sequential(
            *[OptimizedResidualBlock(512) for _ in range(4)] 
        )
        
        # Глубина сети: 5 слоев ConvTranspose2d
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.proj(z)
        x = self.res_blocks(x)  # Добавление residual-соединений
        return self.deconv_layers(x)

# Discriminator 
class EnhancedDiscriminator(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        
        # Количество слоев Conv2d
        self.main = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Дополнительный Conv2d слой
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Финал с адаптивным пулингом
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1, 1, bias=False),
            nn.Flatten(),  # Добавление слоя сглаживания

            # Не используется в WGAN-GP
            #nn.Sigmoid()
        )
        
        # Optimized Residual-блоки в промежуточных слоях
        # (4 оптимизированных блока)
        self.res_blocks = nn.Sequential(
            *[OptimizedResidualBlock(512) for _ in range(4)] 
        )

    def forward(self, img):
        x = self.main(img)
        return x.squeeze()  # Удаление размерности 1

# === GAN END ===
    

# Инициализация
generator = EnhancedGenerator().to(device)
discriminator = EnhancedDiscriminator().to(device)

# criterion BCELoss только для LSTM
# В данной реализации в GAN используется Wasserstein loss
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=G_LEARNING_RATE, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=D_LEARNING_RATE, betas=(0.5, 0.999))


def segment_and_isolate_objects(batch_imgs, save_path, threshold=0.5):
    processed_imgs = []

    for img in batch_imgs:
        # Переводим тензор (C,H,W) -> numpy (H,W,C), uint8
        img_np = (img.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Конвертируем в grayscale для сегментации
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Бинаризация по порогу
        _, binary = cv2.threshold(gray, int(threshold * 255), 255, cv2.THRESH_BINARY)

        # Находим контуры
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Если контуры есть, выбираем самый большой (главный объект)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            # Создаем маску для главного объекта
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [largest_contour], -1, color=255, thickness=-1)
        else:
            # Если контуров нет, маска пустая
            mask = np.zeros_like(gray)

        # Применяем маску к исходному изображению
        masked_img = cv2.bitwise_and(img_np, img_np, mask=mask)

        # На черном фоне (фон уже черный, т.к. masked_img вне маски = 0)
        processed_imgs.append(torch.from_numpy(masked_img).permute(2, 0, 1).float() / 255.0)

    # Собираем батч и сохраняем
    batch_out = torch.stack(processed_imgs).to(batch_imgs.device)
    save_image(batch_out, save_path, nrow=4)
    print(f"Saved segmented batch image to {save_path}")


def denoise_batch(gen_imgs, ksize=5):
    denoised_imgs = []
    for img in gen_imgs:
        # Переводим тензор (C,H,W) в numpy массив (H,W,C) в формате uint8
        img_np = (img.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        # Конвертируем RGB в BGR для OpenCV
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Применяем гауссово размытие
        denoised_bgr = cv2.GaussianBlur(img_bgr, (ksize, ksize), 0)

        # Конвертируем обратно в RGB
        denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)

        # Переводим обратно в тензор и нормируем в [0,1]
        denoised_tensor = torch.from_numpy(denoised_rgb).permute(2, 0, 1).float() / 255.0
        denoised_imgs.append(denoised_tensor)

    return torch.stack(denoised_imgs).to(gen_imgs.device)

# ===
def send_images_in_batches_with_caption(bot: telebot.TeleBot, chat_id: int, batch_imgs: torch.Tensor, caption: str, batch_size=8):
    # Проверка типа
    if not isinstance(chat_id, int):
        raise ValueError(f"Invalid chat_id type: {type(chat_id)}. Must be integer.")
    
    # Проверка существования чата
    try:
        chat = bot.get_chat(chat_id)
    except telebot.apihelper.ApiTelegramException as e:
        print(f"Chat check failed: {e}")
        return
    
    total_imgs = batch_imgs.size(0)
    for start_idx in range(0, total_imgs, batch_size):
        end_idx = min(start_idx + batch_size, total_imgs)
        temp_dir = tempfile.mkdtemp(prefix="telegram_images_")
        media_group = []

        try:
            for i, img_idx in enumerate(range(start_idx, end_idx)):
                img_tensor = batch_imgs[img_idx]
                img_path = os.path.join(temp_dir, f"image_{img_idx}.png")
                save_image(img_tensor, img_path)

                # caption только к первому изображению в группе
                if i == 0:
                    media_group.append(InputMediaPhoto(open(img_path, 'rb'), caption=caption))
                else:
                    media_group.append(InputMediaPhoto(open(img_path, 'rb')))

            bot.send_media_group(chat_id, media_group)

        finally:
            # Закрываем файлы и удаляем временные
            for media in media_group:
                media.media.close()
            for filename in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, filename))
            os.rmdir(temp_dir)

# Визуализирует батч сгенерированных изображений с помощью matplotlib и отправляет в Telegram.
def visualize_batch(gen_imgs, epoch, batch_idx, message_chat_id, epochs):
    # Денормализуем изображения (если они нормализованы)
    gen_imgs = gen_imgs.detach().cpu()
    gen_imgs = (gen_imgs + 1) / 2.0  # Приводим к диапазону [0, 1]

    # Сетка из изображений
    grid_img = utils.make_grid(gen_imgs[:16], nrow=4, normalize=False) # Покажем только первые 16

    # Переводим тензор в numpy и меняем порядок каналов (C, H, W) -> (H, W, C)
    ndarr = grid_img.permute(1, 2, 0).numpy()

    # Конвертируем в uint8
    final_image = (ndarr * 255).astype(np.uint8)

    # Создаем объект io.BytesIO для сохранения изображения в памяти
    img_buffer = io.BytesIO()

    # Сохраняем изображение в формате PNG в буфер
    plt.imsave(img_buffer, final_image, format='png')

    # Перемещаем указатель буфера в начало
    img_buffer.seek(0)

    # Отправляем в TG
    bot.send_photo(message_chat_id, img_buffer, 
                   caption=f"Epoch {epoch}/{epochs} - Batch {batch_idx}")

    # Закрываем буфер
    img_buffer.close()
        
def save_generated_images(epoch, message_chat_id, epochs, batch_idx):
    noise = torch.randn(8, latent_dim, 1, 1, device=device)
    gen_imgs = generator(noise)
    gen_imgs = (gen_imgs + 1) / 2  # денормализация [0,1]

    # Пытаемся избавиться от шума
    gen_imgs = denoise_batch(gen_imgs, ksize=5)
    # Пытаемся сегментировать главный обьект
    segment_and_isolate_objects(gen_imgs, f"generated_segmented_images/segmented_objects_batch_{epoch}.jpg", threshold=SEGMENGT_ISOLATE_THRESHOLD)

    
    save_path = f"{GENERATED_IMAGES_DIR}/epoch_{epoch}_batch_{batch_idx}.png"
    utils.save_image(gen_imgs, save_path, nrow=5)
    
    print(f"Saved generated images to {save_path}")
    send_images_in_batches_with_caption(bot, message_chat_id, gen_imgs, caption=f"[Epoch {epoch}/{epochs}]", batch_size=8)


# Обучение/Переобучение
def relearn(message_chat_id, category_name, epochs):
    bot.send_message(message_chat_id, "Процесс обучения/переобучения занимает время, пожалуйста ждите...")
    
    dataset = datasets.ImageFolder(root=f"./datasets", transform=transform)
    # Получаем индекс у нужной category, затем
    # получаем индексы всех изображений этой category, затем
    # создаем Subset из этих индексов, затем
    # создаем DataLoader из подмножества
    class_idx = dataset.class_to_idx[category_name]
    indices = [i for i, (_, label) in enumerate(dataset.imgs) if label == class_idx]
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)

    lambda_gp = 10  # Обычно 10
    
    for epoch in range(epochs):
        g_running_loss = 0.0
        d_running_loss = 0.0
        num_batches = 0
        gen_steps = 0
        for i, (imgs, _) in enumerate(dataloader):
            real_imgs = imgs.to(device)

            # --- Обучение дискриминатора ---
            optimizer_D.zero_grad()
            z = torch.randn(imgs.size(0), latent_dim, 1, 1, device=device)
            fake_imgs = generator(z)

            real_validity = discriminator(real_imgs)
            fake_validity = discriminator(fake_imgs.detach())
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, device)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            d_loss.backward()
            optimizer_D.step()
            d_running_loss += d_loss.item()
            num_batches += 1

            # --- Обучение генератора (раз в несколько шагов) ---
            if i % 5 == 0:
                optimizer_G.zero_grad()
                gen_imgs = generator(z)
                fake_validity = discriminator(gen_imgs)
                g_loss = -torch.mean(fake_validity)
                g_loss.backward()
                optimizer_G.step()
                g_running_loss += g_loss.item()
                gen_steps += 1

        # Средняя ошибка за эпоху
        avg_d_loss = d_running_loss / num_batches if num_batches > 0 else 0.0
        avg_g_loss = g_running_loss / gen_steps if gen_steps > 0 else 0.0
        print(f"Epoch [{epoch+1}/{epochs}] | D loss: {avg_d_loss:.4f} | G loss: {avg_g_loss:.4f}")

        # Сохраняем изображения каждые LOG_EPOCH_NUM эпох
        if (epoch + 1) % LOG_EPOCH_NUM == 0:
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            gen_imgs = generator(noise)
            visualize_batch(gen_imgs, epoch + 1, i, message_chat_id, epochs)
    
           
# === TG START ===

# Обработчик команды /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn_help = types.KeyboardButton('Помощь')
    markup.add(btn_help)
    
    bot.reply_to(message, 
                 "Привет! Я бот который генерирует изображения. Я был создан для дипломной работы...\n"
                 "Могу ответить на текст или сохранить твои фото для дальнейшего обучения", 
                 reply_markup=markup)

# Обработчик команды /get_categories
@bot.message_handler(commands=['get_categories'])
def get_categories(message):
    base_dir = BASE_DIR  
    if not os.path.exists(base_dir):
        bot.send_message(message.chat.id, "Папка с изображениями не найдена.")
        return

    folders = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]
    if not folders:
        bot.send_message(message.chat.id, "Папок пока нет.")
    else:
        folders_list = "\n".join(folders)
        bot.send_message(message.chat.id, f"Существующие категории:\n{folders_list}")

# Обработчик команды /generate_from_category
@bot.message_handler(commands=['generate_from_category'])
def generate_from_category(message):
    # message.text содержит полный текст команды, например: "/activate 123"
    # Разбиваем на команду и параметры
    parts = message.text.split(maxsplit=1) 
    if len(parts) < 2:
        bot.reply_to(message, "Пожалуйста, укажите category_name, например: /generate_from_category punks")
        return
    args = parts[1]  # Получаем строку с параметрами
    # Разбиваем args по пробелам:
    args_list = args.split()

    # Загрузка конфигурации из файла
    with open('config.json', 'r') as f:
        config = json.load(f)
    category_name = args_list[0]
    epochs = config['datasets'].get(category_name, {}).get('epochs', 16)  # 16 - значение по умолчанию
    print(f"Количество эпох для {category_name}: {epochs}")

    bot.send_message(message.chat.id, "Генерирую изображение...")

    subset_path = os.path.join(BASE_DIR, category_name)
    train_or_load(category_name, subset_path, generator, discriminator, epochs, message.chat.id)

# === LSTM START ===
@bot.message_handler(content_types=['text'])
def handle_text(message):
    # Определяем категорию по тексту через LSTM
    text = message.text.strip()
    seq = torch.tensor([text_to_sequence(text, vocab)], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = lstm_model(seq)
        pred_idx = logits.argmax(1).item()
        pred_cat = idx_to_class[pred_idx]
    bot.send_message(message.chat.id, f'Распознана категория: {pred_cat}\nГенерирую изображение...')
    # Загрузка конфигурации из файла
    with open('config.json', 'r') as f:
        config = json.load(f)
    epochs = config['datasets'].get(pred_cat, {}).get('epochs', 16)

    subset_path = os.path.join(BASE_DIR, pred_cat)
    train_or_load(pred_cat, subset_path, generator, discriminator, epochs, message.chat.id)
# === LSTM END ===


@bot.message_handler(content_types=['photo'])
def save_photo_in_named_folder(message):
    # Получаем текст, отправленный вместе с фото (название папки)
    folder_name = message.caption
    if not folder_name:
        bot.reply_to(message, "Пожалуйста, отправьте фото с подписью - названием папки для сохранения.")
        return

    # Создаем папку, если ее нет
    base_dir = BASE_DIR  # или другой путь
    user_folder = os.path.join(base_dir, folder_name)
    os.makedirs(user_folder, exist_ok=True)

    # Получаем file_id самого большого фото
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    # Формируем уникальное имя файла, чтобы не перезаписывать
    file_ext = os.path.splitext(file_info.file_path)[1]  # например, '.jpg'
    filename = f"{int(time.time())}{file_ext}"
    file_path = os.path.join(user_folder, filename)

    # Сохраняем файл
    with open(file_path, 'wb') as new_file:
        new_file.write(downloaded_file)

    bot.reply_to(message, f"Фото сохранено в категорию '{folder_name}' под именем {filename}")

# Запуск программы
if __name__ == '__main__':
    # === LSTM ===
    # Собираем все тексты и категории
    all_texts = []
    class_names = []
    for cat in os.listdir(BASE_DIR):
        if os.path.isdir(os.path.join(BASE_DIR, cat)):
            class_names.append(cat)
            for file in os.listdir(os.path.join(BASE_DIR, cat)):
                if file.endswith('.txt'):
                    with open(os.path.join(BASE_DIR, cat, file), 'r') as f:
                        all_texts.append(f.read().strip())
    vocab = build_vocab(all_texts)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    dataset = TextCategoryDataset(BASE_DIR, vocab, class_to_idx)
    lstm_model = train_lstm_classifier(dataset, len(vocab), len(class_names), device, epochs=10)
    lstm_model.eval()
    print('LSTM classifier trained')

    # === TG BOT ===    
    print('Запуск бота..')
    bot.polling(none_stop=True)

# == 
