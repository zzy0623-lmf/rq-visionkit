// 后端 API 封装：所有请求统一走 /api，开发期由 Vite 代理到 FastAPI

const BASE = '/api'
const DEPLOY_BASE = '/deploy'

async function req(path, options = {}) {
  const res = await fetch(BASE + path, options)
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail || detail
    } catch (_) {
      /* 忽略非 JSON 响应 */
    }
    throw new Error(detail)
  }
  const ct = res.headers.get('content-type') || ''
  if (ct.includes('application/json')) return res.json()
  return res
}

async function deployReq(path, options = {}) {
  const res = await fetch(DEPLOY_BASE + path, options)
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail || detail
    } catch (_) {
      /* 忽略非 JSON 响应 */
    }
    throw new Error(detail)
  }
  const ct = res.headers.get('content-type') || ''
  if (ct.includes('application/json')) return res.json()
  return res
}

export const api = {
  // 图片列表
  listImages: (status = 'all') => req(`/images?status=${status}`),
  // 原图 URL（交给 <img> 直接加载）
  imageUrl: (id) => `${BASE}/images/${id}/file`,
  // 标注
  getAnnotation: (id) => req(`/annotations/${id}`),
  saveAnnotation: (id, boxes, complete) =>
    req(`/annotations/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ boxes, complete }),
    }),
  // 类别
  getCategories: () => req('/categories'),
  addCategory: (name) =>
    req('/categories', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    }),
  deleteCategory: (id) => req(`/categories/${id}`, { method: 'DELETE' }),
  // 导入 / 导出
  importZip: (file) => {
    const fd = new FormData()
    fd.append('file', file)
    return req('/images/import', { method: 'POST', body: fd })
  },
  exportDataset: () => req('/dataset/export', { method: 'POST' }),
  // 部署控制台（M3）
  deployModelDirs: () => deployReq('/model-dirs'),
  deploy: (payload) =>
    deployReq('', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  deployHealth: (runtimeUrl) => deployReq(`/health?runtime_url=${encodeURIComponent(runtimeUrl)}`),
  deployInfer: (file, runtimeUrl) => {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('runtime_url', runtimeUrl)
    return deployReq('/infer', { method: 'POST', body: fd })
  },
}
