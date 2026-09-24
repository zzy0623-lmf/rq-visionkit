<script setup>
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import { api } from './api.js'
import CanvasAnnotator from './components/CanvasAnnotator.vue'
import DeployPanel from './components/DeployPanel.vue'

const tab = ref('annotate') // 'annotate' | 'deploy'
const view = ref('grid') // 'grid' | 'annotate'
const images = ref([])
const filter = ref('all')
const classes = ref([])
const error = ref('')

// 当前标注对象
const current = ref(null) // 图片元数据
const boxes = ref([])
const complete = ref(false)
const selectedClassId = ref(0)
const newCls = ref('')

const progress = computed(() => {
  const total = images.value.length
  const labeled = images.value.filter((i) => i.box_count > 0 || i.complete).length
  return { total, labeled, pct: total ? Math.round((labeled / total) * 100) : 0 }
})

const filtered = computed(() => {
  if (filter.value === 'all') return images.value
  const isLabeled = (i) => i.box_count > 0 || i.complete
  return images.value.filter((i) => (filter.value === 'labeled' ? isLabeled(i) : !isLabeled(i)))
})

async function loadAll() {
  const [imgs, cats] = await Promise.all([api.listImages(filter.value), api.getCategories()])
  images.value = imgs.items
  classes.value = cats.classes
}

async function openAnnotator(img) {
  const ann = await api.getAnnotation(img.id)
  current.value = img
  boxes.value = ann.boxes || []
  complete.value = !!ann.complete
  selectedClassId.value = classes.value[0]?.id ?? 0
  view.value = 'annotate'
}

async function persist() {
  if (!current.value) return
  try {
    await api.saveAnnotation(current.value.id, boxes.value, complete.value)
    error.value = ''
  } catch (e) {
    error.value = e.message
  }
}

async function onAddBox(box) {
  boxes.value.push(box)
  await persist()
}

async function onDeleteBox(i) {
  boxes.value.splice(i, 1)
  await persist()
}

function onChangeClass(id) {
  selectedClassId.value = id
}

async function toggleComplete() {
  complete.value = !complete.value
  await persist()
}

function currentIndex() {
  return filtered.value.findIndex((i) => i.id === current.value.id)
}

async function goto(step) {
  const list = filtered.value
  const idx = currentIndex()
  const next = list[idx + step]
  if (next) await openAnnotator(next)
}

async function addCategory(name) {
  if (!name) return
  await api.addCategory(name)
  await loadAll()
}

async function removeCategory(id) {
  if (!confirm('删除该类别？若已被标注框引用将失败。')) return
  try {
    await api.deleteCategory(id)
    await loadAll()
  } catch (e) {
    error.value = e.message
  }
}

async function onImportZip(e) {
  const file = e.target.files?.[0]
  if (!file) return
  try {
    await api.importZip(file)
    await loadAll()
    e.target.value = ''
  } catch (err) {
    error.value = err.message
  }
}

async function onExport() {
  try {
    const blob = await api.exportDataset()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'yolo_dataset.zip'
    a.click()
    URL.revokeObjectURL(url)
  } catch (err) {
    error.value = err.message
  }
}

// 快捷键：N 下一张 / P 上一张（仅标注页，且不聚焦输入框）
function onKeyDown(e) {
  if (view.value !== 'annotate') return
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return
  if (e.key === 'n' || e.key === 'N') goto(1)
  if (e.key === 'p' || e.key === 'P') goto(-1)
}

onMounted(() => {
  loadAll()
  window.addEventListener('keydown', onKeyDown)
})
onBeforeUnmount(() => window.removeEventListener('keydown', onKeyDown))
</script>

<template>
  <div class="app">
    <header class="bar">
      <h1>RQ-VisionKit</h1>
      <nav class="main-nav">
        <button :class="{ active: tab === 'annotate' }" @click="tab = 'annotate'">采集标注</button>
        <button :class="{ active: tab === 'deploy' }" @click="tab = 'deploy'">模型部署</button>
      </nav>
      <div class="bar-actions">
        <template v-if="tab === 'annotate'">
          <label class="btn">
            导入 zip
            <input type="file" accept=".zip" hidden @change="onImportZip" />
          </label>
          <button class="btn" @click="onExport">导出数据集</button>
        </template>
      </div>
    </header>

    <p v-if="error" class="err">{{ error }}</p>

    <!-- 部署控制台（M3） -->
    <section v-if="tab === 'deploy'">
      <DeployPanel />
    </section>

    <!-- 网格列表视图 -->
    <section v-else-if="view === 'grid'" class="grid-view">
      <div class="toolbar">
        <div class="tabs">
          <button :class="{ active: filter === 'all' }" @click="filter = 'all'">全部</button>
          <button :class="{ active: filter === 'unlabeled' }" @click="filter = 'unlabeled'">未标注</button>
          <button :class="{ active: filter === 'labeled' }" @click="filter = 'labeled'">已标注</button>
        </div>
        <div class="progress">
          进度 {{ progress.labeled }}/{{ progress.total }}
          <div class="pbar"><div class="pfill" :style="{ width: progress.pct + '%' }"></div></div>
        </div>
      </div>

      <div v-if="filtered.length === 0" class="empty">暂无图片，请先「导入 zip」。</div>
      <div class="grid">
        <div
          v-for="img in filtered"
          :key="img.id"
          class="cell"
          :class="{ done: img.box_count > 0 || img.complete }"
          @click="openAnnotator(img)"
        >
          <img :src="api.imageUrl(img.id)" :alt="img.filename" loading="lazy" />
          <div class="cell-meta">
            <span class="cell-name">{{ img.filename }}</span>
            <span class="cell-tag">{{ img.box_count > 0 || img.complete ? img.box_count + ' 框' : '未标注' }}</span>
          </div>
        </div>
      </div>
    </section>

    <!-- 标注详情视图 -->
    <section v-else-if="current" class="anno-view">
      <div class="anno-head">
        <button class="btn" @click="view = 'grid'">← 返回列表</button>
        <span class="anno-title">{{ current.filename }}</span>
        <div class="anno-nav">
          <button class="btn" @click="goto(-1)">上一张 (P)</button>
          <button class="btn" @click="goto(1)">下一张 (N)</button>
        </div>
      </div>

      <div class="anno-body">
        <CanvasAnnotator
          class="anno-canvas"
          :image-url="api.imageUrl(current.id)"
          :image-width="current.width"
          :image-height="current.height"
          :boxes="boxes"
          :classes="classes"
          :selected-class-id="selectedClassId"
          @add-box="onAddBox"
          @delete-box="onDeleteBox"
          @change-class="onChangeClass"
        />

        <aside class="panel">
          <h3>类别（数字键 1-9 切换）</h3>
          <ul class="cls-list">
            <li
              v-for="c in classes"
              :key="c.id"
              :class="{ sel: c.id === selectedClassId }"
              @click="selectedClassId = c.id"
            >
              <span class="cls-key">{{ classes.indexOf(c) + 1 }}</span>
              {{ c.name }}
              <button class="del" @click.stop="removeCategory(c.id)">×</button>
            </li>
          </ul>
          <div class="cls-add">
            <input v-model="newCls" placeholder="新类别名" @keyup.enter="addCategory(newCls); newCls = ''" />
            <button class="btn" @click="addCategory(newCls); newCls = ''">添加</button>
          </div>

          <label class="done-toggle">
            <input type="checkbox" :checked="complete" @change="toggleComplete" />
            标注完成
          </label>

          <p class="hint">拖拽画框 · 点击选中 · Del 删框 · N/P 切换图片 · 数字键选类别（自动保存）</p>
        </aside>
      </div>
    </section>
  </div>
</template>
