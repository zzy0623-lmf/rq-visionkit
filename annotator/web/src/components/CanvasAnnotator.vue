<script setup>
import { ref, onMounted, onBeforeUnmount, watch } from 'vue'

// 自研 Canvas 标注层：拖拽画框 + 选中 + 删除 + 数字键切类别。
// 框坐标统一使用图片原始像素 (x1,y1,x2,y2)，与后端约定一致。
const props = defineProps({
  imageUrl: { type: String, required: true },
  imageWidth: { type: Number, required: true },
  imageHeight: { type: Number, required: true },
  boxes: { type: Array, default: () => [] },
  classes: { type: Array, default: () => [] },
  selectedClassId: { type: Number, default: 0 },
})

const emit = defineEmits(['add-box', 'delete-box', 'change-class'])

const canvas = ref(null)
const wrap = ref(null)
let selectedIndex = null // 当前选中框下标（-1 表示无）
let drawing = null // 拖拽中的临时框 {x1,y1,x2,y2}

// 图片 → canvas 的统一缩放系数（canvas 与图片同宽高比，故 x/y 缩放一致）
function scale() {
  return canvas.value.width / props.imageWidth
}

function toImage(e) {
  const rect = canvas.value.getBoundingClientRect()
  return {
    x: (e.clientX - rect.left) / scale(),
    y: (e.clientY - rect.top) / scale(),
  }
}

function resize() {
  if (!wrap.value || !props.imageWidth) return
  const w = wrap.value.clientWidth
  const h = (w * props.imageHeight) / props.imageWidth
  canvas.value.width = w
  canvas.value.height = h
  draw()
}

function normBox(b) {
  return {
    x1: Math.min(b.x1, b.x2),
    y1: Math.min(b.y1, b.y2),
    x2: Math.max(b.x1, b.x2),
    y2: Math.max(b.y1, b.y2),
  }
}

function draw() {
  const ctx = canvas.value.getContext('2d')
  const s = scale()
  ctx.clearRect(0, 0, canvas.value.width, canvas.value.height)
  const color = (i) => (i === selectedIndex ? '#ff5a5f' : '#22c55e')

  props.boxes.forEach((b, i) => {
    ctx.strokeStyle = color(i)
    ctx.lineWidth = 2
    ctx.strokeRect(b.x1 * s, b.y1 * s, (b.x2 - b.x1) * s, (b.y2 - b.y1) * s)
  })
  if (drawing) {
    const b = normBox(drawing)
    ctx.strokeStyle = '#ff5a5f'
    ctx.lineWidth = 2
    ctx.setLineDash([6, 4])
    ctx.strokeRect(b.x1 * s, b.y1 * s, (b.x2 - b.x1) * s, (b.y2 - b.y1) * s)
    ctx.setLineDash([])
  }
}

function hitTest(p) {
  // 命中测试：返回最上层（最后绘制）命中的框下标
  for (let i = props.boxes.length - 1; i >= 0; i--) {
    const b = props.boxes[i]
    if (p.x >= b.x1 && p.x <= b.x2 && p.y >= b.y1 && p.y <= b.y2) return i
  }
  return -1
}

function onDown(e) {
  e.preventDefault()
  const p = toImage(e)
  const hit = hitTest(p)
  selectedIndex = hit
  if (hit === -1) {
    // 空白处开始画新框
    drawing = { x1: p.x, y1: p.y, x2: p.x, y2: p.y }
  }
  draw()
}

function onMove(e) {
  if (!drawing) return
  const p = toImage(e)
  drawing.x2 = p.x
  drawing.y2 = p.y
  draw()
}

function onUp() {
  if (drawing) {
    const b = normBox(drawing)
    if (b.x2 - b.x1 > 2 && b.y2 - b.y1 > 2) {
      emit('add-box', b)
    }
    drawing = null
    draw()
  }
}

function onKey(e) {
  if (e.key === 'Delete' || e.key === 'Backspace') {
    if (selectedIndex !== null) {
      emit('delete-box', selectedIndex)
      selectedIndex = null
      draw()
    }
    return
  }
  // 数字键 1-9 切类别
  const n = parseInt(e.key, 10)
  if (n >= 1 && n <= 9 && n - 1 < props.classes.length) {
    emit('change-class', props.classes[n - 1].id)
  }
}

let ro = null
onMounted(() => {
  resize()
  ro = new ResizeObserver(resize)
  ro.observe(wrap.value)
  window.addEventListener('keydown', onKey)
})

onBeforeUnmount(() => {
  ro?.disconnect()
  window.removeEventListener('keydown', onKey)
})

// boxes / 类别变化时重绘（选中下标失效则清空）
watch(
  () => [props.boxes, props.imageUrl, props.selectedClassId],
  () => {
    if (selectedIndex !== null && selectedIndex >= props.boxes.length) selectedIndex = null
    drawing = null
    draw()
  },
)
</script>

<template>
  <div ref="wrap" class="cv-wrap">
    <canvas
      ref="canvas"
      class="cv"
      @pointerdown="onDown"
      @pointermove="onMove"
      @pointerup="onUp"
      @pointercancel="onUp"
    ></canvas>
    <img class="cv-img" :src="imageUrl" alt="标注原图" />
  </div>
</template>

<style scoped>
.cv-wrap {
  position: relative;
  width: 100%;
  line-height: 0;
  touch-action: none;
}
.cv {
  width: 100%;
  height: auto;
  cursor: crosshair;
  display: block;
}
/* 隐藏原图，仅用于预加载/缓存；Canvas 上叠加绘制 */
.cv-img {
  display: none;
}
</style>
