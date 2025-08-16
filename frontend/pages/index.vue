<template>
  <div class="min-h-screen bg-gray-50 py-8">
    <div class="max-w-4xl mx-auto px-4">
      <h1 class="text-3xl font-bold text-center mb-8 text-gray-800">
        Image Text Translator
      </h1>
      
      <!-- Upload Section -->
      <div class="bg-white rounded-lg shadow-md p-6 mb-6">
        <h2 class="text-xl font-semibold mb-4">Upload Image</h2>
        <div class="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
          <input 
            ref="fileInput"
            type="file" 
            accept="image/*" 
            @change="handleFileUpload"
            class="hidden"
          >
          <button 
            @click="$refs.fileInput.click()"
            class="bg-blue-500 hover:bg-blue-600 text-white px-6 py-3 rounded-lg font-medium"
          >
            Choose Image
          </button>
          <p class="mt-2 text-gray-500">or drag and drop an image here</p>
        </div>
        
        <!-- Preview -->
        <div v-if="selectedImage" class="mt-4">
          <img :src="selectedImage" alt="Preview" class="max-w-full h-auto rounded-lg">
        </div>
      </div>

      <!-- Language Selection -->
      <div class="bg-white rounded-lg shadow-md p-6 mb-6">
        <h2 class="text-xl font-semibold mb-4">Language Settings</h2>
        <div class="grid grid-cols-1 md:grid-cols-8 gap-4">
          <div class="md:col-span-2">
            <label class="block text-sm font-medium text-gray-700 mb-2">
              Source Language
            </label>
            <select 
              v-model="sourceLanguage" 
              class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="en">English</option>
              <option value="zh">Chinese</option>
            </select>
          </div>
          <div class="md:col-span-2">
            <label class="block text-sm font-medium text-gray-700 mb-2">
              Target Language
            </label>
            <select 
              v-model="targetLanguage" 
              class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="en">English</option>
              <option value="zh">Chinese</option>
            </select>
          </div>
          <div class="md:col-span-3">
            <label class="block text-sm font-medium text-gray-700 mb-2">
              Target Font
            </label>
            <select 
              v-model="selectedFont" 
              class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="/usr/share/fonts/truetype/arphic/uming.ttc">AR PL UMing</option>
              <option value="/usr/share/fonts/truetype/arphic/ukai.ttc">AR PL UKai</option>
              <option value="/usr/share/fonts/truetype/wqy/wqy-microhei.ttc">WenQuanYi Micro Hei</option>
              <option value="/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc">WenQuanYi Zen Hei</option>
              <option value="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf">Sans</option>
            </select>
          </div>
          <div class="md:col-span-1">
            <label class="block text-sm font-medium text-gray-700 mb-2">
              Scale Factor
            </label>
            <input 
              v-model.number="scaleFactor" 
              type="number" 
              step="0.1" 
              min="0.1" 
              max="3.0" 
              class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="1.0"
            >
          </div>
        </div>
      </div>

      <!-- Process Button -->
      <div class="text-center mb-6">
        <button 
          @click="processImage"
          :disabled="!selectedFile || processing"
          class="bg-green-500 hover:bg-green-600 disabled:bg-gray-400 text-white px-8 py-3 rounded-lg font-medium text-lg"
        >
          {{ processing ? 'Processing...' : 'Translate Image' }}
        </button>
      </div>

      <!-- Results -->
      <div v-if="processedImage" class="bg-white rounded-lg shadow-md p-6">
        <h2 class="text-xl font-semibold mb-4">Translated Image</h2>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h3 class="text-lg font-medium mb-2">Original</h3>
            <img :src="selectedImage" alt="Original" class="w-full h-auto rounded-lg">
          </div>
          <div>
            <h3 class="text-lg font-medium mb-2">Translated</h3>
            <img :src="processedImage" alt="Translated" class="w-full h-auto rounded-lg">
          </div>
        </div>
        
        <!-- Download Button -->
        <div class="text-center mt-4">
          <a 
            :href="processedImage" 
            download="translated_image.png"
            class="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-lg font-medium inline-block"
          >
            Download Translated Image
          </a>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const selectedFile = ref(null)
const selectedImage = ref(null)
const sourceLanguage = ref('en')
const targetLanguage = ref('zh')
const selectedFont = ref('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
const scaleFactor = ref(1.0) // Default scale factor
const processing = ref(false)
const processedImage = ref(null)

const handleFileUpload = (event) => {
  const file = event.target.files[0]
  if (file) {
    selectedFile.value = file
    const reader = new FileReader()
    reader.onload = (e) => {
      selectedImage.value = e.target.result
    }
    reader.readAsDataURL(file)
  }
}

const processImage = async () => {
  if (!selectedFile.value) return
  
  processing.value = true
  
  try {
    const formData = new FormData()
    formData.append('image', selectedFile.value)
    formData.append('source_lang', sourceLanguage.value)
    formData.append('target_lang', targetLanguage.value)
    formData.append('scale_factor', scaleFactor.value.toString())
    
    // Only add font parameter if target language is Chinese
    if (targetLanguage.value === 'zh') {
      formData.append('font_path', selectedFont.value)
    }
    
    const response = await $fetch('http://localhost:8000/api/translate-image', {
      method: 'POST',
      body: formData
    })
    
    processedImage.value = `data:image/png;base64,${response.translated_image}`
  } catch (error) {
    console.error('Error processing image:', error)
    alert('Error processing image. Please try again.')
  } finally {
    processing.value = false
  }
}
</script>