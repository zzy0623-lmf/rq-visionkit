<script setup>
import { ref, computed, onMounted } from 'vue'
import { api } from '../api.js'

const DEFAULT_CLASSES = 'crazing\ninclusion\npatches\npitted_surface\nrolled-in_scale\nscratches'

// 部署表单
const modelDirs = ref([])
const form = ref({
  model_dir: '',
  model_name: 'yolox_neudet',
  task: 'detect',
  input_size: 320,
  conf_thres: 0.01,
  nms_thres: 0.65,
  quant: 'fp32',
  classes_text: DEFAULT_CLASSES,
  transport: 'local',
  host: '127.0.0.1',
  username: 'root',
  port: 22,
  model_dir_target: '',
  config_path: '',
  runtime_url: 'http://127.0.0.1:8001',
})

const deploying = ref(false)
const result = ref(null)
const error = ref('')

// 结果看板
const inferring = ref(false)
const board = ref(null) // detect: {boxes, overlay,...} | classify: {classes, overlay,...}
const health = ref(null)

const classes = computed(() =>
  form.value.classes_text.split('\n').map((s) => s.trim()).filter(Boolean)
)

function buildPayload() {
  return {
    model_dir: form.value.model_dir,
    model_name: form.value.model_name,
    task: form.value.task,
    input_size: Number(form.value.input_size),
    conf_thres: Number(form.value.conf_thres),
    nms_thres: Number(form.value.nms_thres),
    quant: form.value.quant,
    classes: classes.value,
    target: {
      transport: form.value.transport,
      host: form.value.host,
      username: form.value.username,
      port: Number(form.value.port),
      model_dir: form.value.model_dir_target,
      config_path: form.value.config_path,
      runtime_url: form.value.runtime_url,
    },
  }
}

async function loadModelDirs() {
  try {
    const r = await api.deployModelDirs()
    modelDirs.value = r.dirs
    if (!form.value.model_dir && r.dirs.length) form.value.model_dir = r.dirs[0].path
  } catch (e) {
    error.value = e.message
  }
}

async function onDeploy() {
  deploying.value = true
  result.value = null
  error.value = ''
  try {
    result.value = await api.deploy(buildPayload())
  } catch (e) {
    error.value = e.message
  } finally {
    deploying.value = false
  }
}

async function onHealth() {
  error.value = ''
  try {
    health.value = await api.deployHealth(form.value.runtime_url)
  } catch (e) {
    error.value = e.message
  }
}

async function onInfer(e) {
  const file = e.target.files?.[0]
  if (!file) return
  inferring.value = true
  board.value = null
  error.value = ''
  try {
    board.value = await api.deployInfer(file, form.value.runtime_url)
  } catch (err) {
    error.value = err.message
  } finally {
    inferring.value = false
    e.target.value = ''
  }
}

onMounted(loadModelDirs)
</script>

<template>
  <div class="deploy">
    <p v-if="error" class="err">{{ error }}</p>

    <div class="deploy-cols">
      <!-- 部署配置表单 -->
      <section class="card">
        <h2>部署配置</h2>

        <label class="field">
          <span>模型目录</span>
          <select v-model="form.model_dir">
            <option v-for="d in modelDirs" :key="d.path" :value="d.path">
              {{ d.name }}（{{ d.param_count }} 个模型）
            </option>
          </select>
          <input v-model="form.model_dir" placeholder="或手动输入模型文件目录" />
        </label>

        <div class="row">
          <label class="field">
            <span>任务类型</span>
            <select v-model="form.task">
              <option value="detect">检测（YOLOX）</option>
              <option value="classify">分类（softmax）</option>
            </select>
          </label>
          <label class="field">
            <span>模型名</span>
            <input v-model="form.model_name" />
          </label>
          <label class="field">
            <span>输入尺寸</span>
            <input v-model.number="form.input_size" type="number" />
          </label>
        </div>

        <div class="row">
          <label class="field">
            <span>置信度阈值</span>
            <input v-model.number="form.conf_thres" type="number" step="0.01" />
          </label>
          <label class="field">
            <span>NMS 阈值</span>
            <input v-model.number="form.nms_thres" type="number" step="0.05" />
          </label>
          <label class="field">
            <span>量化档位</span>
            <select v-model="form.quant">
              <option value="fp32">fp32</option>
              <option value="int8">int8</option>
            </select>
          </label>
        </div>

        <label class="field">
          <span>类别（每行一个）</span>
          <textarea v-model="form.classes_text" rows="4"></textarea>
        </label>

        <h3>目标设备</h3>
        <div class="row">
          <label class="field">
            <span>传输方式</span>
            <select v-model="form.transport">
              <option value="local">本地（PC 仿真）</option>
              <option value="ssh">SSH（RK3506）</option>
            </select>
          </label>
          <label class="field">
            <span>运行时地址</span>
            <input v-model="form.runtime_url" />
          </label>
        </div>

        <template v-if="form.transport === 'ssh'">
          <div class="row">
            <label class="field">
              <span>主机</span>
              <input v-model="form.host" />
            </label>
            <label class="field">
              <span>用户名</span>
              <input v-model="form.username" />
            </label>
            <label class="field">
              <span>端口</span>
              <input v-model.number="form.port" type="number" />
            </label>
          </div>
          <label class="field">
            <span>设备模型目录</span>
            <input v-model="form.model_dir_target" placeholder="如 /data/model" />
          </label>
          <label class="field">
            <span>设备 config.yaml 路径</span>
            <input v-model="form.config_path" placeholder="如 /etc/rq-visionkit/config.yaml" />
          </label>
        </template>
        <template v-else>
          <label class="field">
            <span>本地模型目录</span>
            <input v-model="form.model_dir_target" placeholder="留空用默认 runtime/pc_sim/deployed_model" />
          </label>
          <label class="field">
            <span>本地 config.yaml 路径</span>
            <input v-model="form.config_path" placeholder="留空用默认 runtime/pc_sim/config.yaml" />
          </label>
        </template>

        <button class="btn primary" :disabled="deploying" @click="onDeploy">
          {{ deploying ? '部署中…' : '一键部署' }}
        </button>

        <div v-if="result" class="ok-box">
          <div>已部署：{{ result.model_name }} v{{ result.version }}</div>
          <div>下发文件：{{ (result.files || []).join('、') }}</div>
          <div>reload：{{ JSON.stringify(result.reload) }}</div>
        </div>
      </section>

      <!-- 结果看板 -->
      <section class="card">
        <h2>结果看板</h2>
        <div class="row">
          <label class="btn">
            上传图片推理
            <input type="file" accept="image/*" hidden @change="onInfer" />
          </label>
          <button class="btn" @click="onHealth">查看设备状态</button>
        </div>

        <div v-if="health" class="health">
          模型：{{ health.model_name }} v{{ health.version }} ·
          量化：{{ health.quant }} · 类别数：{{ health.num_classes }}
        </div>

        <div v-if="inferring" class="muted">推理中…</div>

        <div v-if="board" class="board">
          <img class="board-img" :src="board.overlay" alt="结果图" />
          <div class="metrics">
            <div class="metric"><b>{{ board.fps ?? '—' }}</b><span>FPS</span></div>
            <div class="metric"><b>{{ board.inference_ms }}ms</b><span>单帧时延</span></div>
            <div class="metric"><b>{{ board.mem_kb }}KB</b><span>内存占用</span></div>
          </div>

          <!-- 分类结果：top-k 类别 + 概率 -->
          <table v-if="board.task === 'classify'" class="box-table">
            <thead>
              <tr><th>排名</th><th>类别</th><th>概率</th></tr>
            </thead>
            <tbody>
              <tr v-for="(c, i) in board.classes" :key="i">
                <td>{{ i + 1 }}</td>
                <td>{{ c.class_name }}</td>
                <td>{{ c.score.toFixed(4) }}</td>
              </tr>
            </tbody>
          </table>

          <!-- 检测结果：框列表 -->
          <table v-else-if="board.boxes.length" class="box-table">
            <thead>
              <tr><th>类别</th><th>置信度</th><th>坐标 (x1,y1,x2,y2)</th></tr>
            </thead>
            <tbody>
              <tr v-for="(b, i) in board.boxes" :key="i">
                <td>{{ b.class_name }}</td>
                <td>{{ b.score.toFixed(3) }}</td>
                <td>{{ Math.round(b.x1) }},{{ Math.round(b.y1) }},{{ Math.round(b.x2) }},{{ Math.round(b.y2) }}</td>
              </tr>
            </tbody>
          </table>
          <p v-else class="muted">未检测到目标。</p>
        </div>
      </section>
    </div>
  </div>
</template>

<style scoped>
.deploy-cols {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  align-items: start;
}
.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px;
}
.card h2 {
  font-size: 15px;
  margin: 0 0 12px;
}
.card h3 {
  font-size: 13px;
  margin: 16px 0 8px;
  color: var(--muted);
}
.field {
  display: block;
  margin-bottom: 10px;
}
.field span {
  display: block;
  font-size: 12px;
  color: var(--muted);
  margin-bottom: 4px;
}
.field input,
.field select,
.field textarea {
  width: 100%;
  padding: 7px 8px;
  border: 1px solid var(--line);
  border-radius: 6px;
  font-size: 13px;
  font-family: inherit;
}
.field textarea {
  resize: vertical;
}
.row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
.row .field {
  flex: 1;
  min-width: 100px;
}
.btn.primary {
  background: var(--brand);
  color: #fff;
  border-color: var(--brand);
  margin-top: 8px;
}
.btn.primary:hover {
  background: var(--brand-dark);
}
.ok-box {
  margin-top: 12px;
  background: #eff6ff;
  border: 1px solid #bfdbfe;
  border-radius: 6px;
  padding: 10px 12px;
  font-size: 12px;
  line-height: 1.7;
}
.health {
  margin-top: 8px;
  font-size: 13px;
  color: var(--muted);
}
.board {
  margin-top: 12px;
}
.board-img {
  width: 100%;
  border-radius: 8px;
  background: #000;
  display: block;
}
.metrics {
  display: flex;
  gap: 12px;
  margin: 12px 0;
}
.metric {
  flex: 1;
  background: var(--bg);
  border-radius: 6px;
  padding: 10px;
  text-align: center;
}
.metric b {
  display: block;
  font-size: 18px;
}
.metric span {
  font-size: 12px;
  color: var(--muted);
}
.box-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}
.box-table th,
.box-table td {
  border: 1px solid var(--line);
  padding: 5px 6px;
  text-align: left;
}
.muted {
  color: var(--muted);
}
@media (max-width: 900px) {
  .deploy-cols {
    grid-template-columns: 1fr;
  }
}
</style>
