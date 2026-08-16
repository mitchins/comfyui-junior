// IndexedDB Client Storage for LAN Appliance
const DB_NAME = "ImagineDB";
const DB_VERSION = 1;
const STORE_NAME = "generations";
const MAX_HISTORY = 20;

class ImageStorage {
  constructor() {
    this.db = null;
  }

  async init() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);
      request.onupgradeneeded = (e) => {
        const db = e.target.result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          const store = db.createObjectStore(STORE_NAME, { keyPath: "id" });
          store.createIndex("timestamp", "timestamp", { unique: false });
        }
      };
      request.onsuccess = (e) => {
        this.db = e.target.result;
        resolve(this.db);
      };
      request.onerror = (e) => reject(e.target.error);
    });
  }

  async saveGeneration(item) {
    if (!this.db) await this.init();
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction([STORE_NAME], "readwrite");
      const store = tx.objectStore(STORE_NAME);
      store.put(item);

      tx.oncomplete = async () => {
        await this.evictOldest();
        resolve(item);
      };
      tx.onerror = (e) => reject(e.target.error);
    });
  }

  async getAllRecent() {
    if (!this.db) await this.init();
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction([STORE_NAME], "readonly");
      const store = tx.objectStore(STORE_NAME);
      const index = store.index("timestamp");
      const items = [];
      const cursorReq = index.openCursor(null, "prev"); // newest first

      cursorReq.onsuccess = (e) => {
        const cursor = e.target.result;
        if (cursor && items.length < MAX_HISTORY) {
          items.push(cursor.value);
          cursor.continue();
        } else {
          resolve(items);
        }
      };
      cursorReq.onerror = (e) => reject(e.target.error);
    });
  }

  async evictOldest() {
    return new Promise((resolve) => {
      const tx = this.db.transaction([STORE_NAME], "readwrite");
      const store = tx.objectStore(STORE_NAME);
      const index = store.index("timestamp");
      const keys = [];

      const cursorReq = index.openKeyCursor(null, "prev");
      cursorReq.onsuccess = (e) => {
        const cursor = e.target.result;
        if (cursor) {
          keys.push(cursor.primaryKey);
          cursor.continue();
        } else {
          if (keys.length > MAX_HISTORY) {
            const toDelete = keys.slice(MAX_HISTORY);
            for (const key of toDelete) {
              store.delete(key);
            }
          }
          resolve();
        }
      };
    });
  }
}

const storage = new ImageStorage();

function b64ToBlob(b64Data, contentType = "image/png") {
  const byteCharacters = atob(b64Data);
  const byteArrays = [];
  const sliceSize = 512;

  for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {
    const slice = byteCharacters.slice(offset, offset + sliceSize);
    const byteNumbers = new Array(slice.length);
    for (let i = 0; i < slice.length; i++) {
      byteNumbers[i] = slice.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    byteArrays.push(byteArray);
  }

  return new Blob(byteArrays, { type: contentType });
}

function slugify(text) {
  return text
    .toString()
    .toLowerCase()
    .trim()
    .replace(/\s+/g, "-")
    .replace(/[^\w\-]+/g, "")
    .replace(/\-\-+/g, "-")
    .slice(0, 30);
}

// Friendly formatting for human-readable reasons
const REASON_LABELS = {
  sexual_content: "sexual or explicit content",
  sexual: "suggestive content",
  nudity: "nudity",
  violence_gore: "violence or gore",
  substances: "drugs or controlled substances",
  disturbing_content: "disturbing content",
  disturbing: "disturbing content",
  fetish_content: "inappropriate content",
  fetish: "inappropriate content",
};

function formatRestrainedReason(reasons) {
  if (!reasons || !Array.isArray(reasons) || reasons.length === 0) {
    return "Reason: content policy";
  }
  const formatted = reasons
    .map((r) => REASON_LABELS[r] || r.replace(/_/g, " "))
    .filter((v, i, a) => a.indexOf(v) === i)
    .join(", ");
  return `Reason: ${formatted}`;
}

document.addEventListener("alpine:init", () => {
  Alpine.data("imagineApp", () => ({
    aspect: "square",
    quality: "normal",
    prompt: "",
    isGenerating: false,
    currentImage: null,
    errorMessage: null,
    errorDetails: null,
    showErrorDetails: false,
    recentList: [],
    activeBlobUrls: new Set(),

    sizes: {
      normal: {
        portrait: "768x1024",
        square: "768x768",
        landscape: "1024x768",
      },
      high: {
        portrait: "1024x1344",
        square: "1024x1024",
        landscape: "1344x1024",
      },
    },

    async init() {
      try {
        await storage.init();
        await this.loadRecentHistory();
      } catch (err) {
        console.warn("IndexedDB init failed:", err);
      }
    },

    async loadRecentHistory() {
      const stored = await storage.getAllRecent();
      this.recentList = stored.map((item) => {
        const blobUrl = URL.createObjectURL(item.blob);
        this.activeBlobUrls.add(blobUrl);
        return {
          ...item,
          blobUrl,
        };
      });

      if (this.recentList.length > 0 && !this.currentImage) {
        this.currentImage = this.recentList[0];
      }
    },

    handleKeydown(e) {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        this.generate();
      }
    },

    async generate() {
      const cleanPrompt = this.prompt.trim();
      if (!cleanPrompt || this.isGenerating) return;

      this.isGenerating = true;
      this.errorMessage = null;
      this.errorDetails = null;
      this.showErrorDetails = false;

      const targetSize = this.sizes[this.quality][this.aspect];

      try {
        const resp = await fetch("/v1/images/generations", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt: cleanPrompt,
            model: "flux2-klein-4b-safe",
            size: targetSize,
            response_format: "b64_json",
          }),
        });

        const data = await resp.json();

        if (!resp.ok) {
          console.debug("[API Diagnostics]", data);
          const err = data.error || {};
          const code = err.code || "";
          const reasons = data.reasons || err.safety_reasons || [];
          
          if (code === "content_policy_violation" || code === "safety_route_required") {
            this.errorMessage = "That idea isn't available here. Try changing the picture a little.";
            this.errorDetails = formatRestrainedReason(reasons);
          } else {
            this.errorMessage = "That picture couldn't be made right now. Please try again.";
            this.errorDetails = "Reason: service error";
          }
          return;
        }

        const b64Json = data.data?.[0]?.b64_json;
        if (!b64Json) {
          throw new Error("No image data received in response");
        }

        const blob = b64ToBlob(b64Json, "image/png");
        const blobUrl = URL.createObjectURL(blob);
        this.activeBlobUrls.add(blobUrl);

        const newRecord = {
          id: `img_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
          timestamp: Date.now(),
          prompt: cleanPrompt,
          aspect: this.aspect,
          quality: this.quality,
          size: targetSize,
          blob: blob,
        };

        await storage.saveGeneration(newRecord);

        const viewItem = {
          ...newRecord,
          blobUrl,
        };

        this.currentImage = viewItem;
        this.recentList.unshift(viewItem);
        if (this.recentList.length > MAX_HISTORY) {
          const evicted = this.recentList.pop();
          if (evicted?.blobUrl) {
            URL.revokeObjectURL(evicted.blobUrl);
            this.activeBlobUrls.delete(evicted.blobUrl);
          }
        }
      } catch (err) {
        console.error("Generation error:", err);
        this.errorMessage = "That picture couldn't be made right now. Check connection and try again.";
        this.errorDetails = "Reason: network error";
      } finally {
        this.isGenerating = false;
      }
    },

    selectRecent(item) {
      this.currentImage = item;
      this.prompt = item.prompt;
      if (item.aspect) this.aspect = item.aspect;
      if (item.quality) this.quality = item.quality;
      this.errorMessage = null;
    },

    makeAnother() {
      this.generate();
    },

    editPrompt() {
      const textarea = document.getElementById("prompt-input");
      if (textarea) {
        textarea.focus();
        textarea.setSelectionRange(textarea.value.length, textarea.value.length);
      }
    },

    downloadImage() {
      if (!this.currentImage || !this.currentImage.blobUrl) return;
      const slug = slugify(this.currentImage.prompt || "imagine");
      const filename = `imagine-${slug}-${Date.now()}.png`;

      const link = document.createElement("a");
      link.href = this.currentImage.blobUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    },
  }));
});
