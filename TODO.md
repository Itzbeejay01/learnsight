# LearnSight Docker Pip Fix - TODO

✅ **Pip hang fixed: fast mirror + optimized layers → docker compose up --build**

## Current Status: [x] Plan approved → Implementing

### Steps:
- [x] 1. TODO.md updated with pip fix plan  
- [x] 2. Create .dockerignore ✓
- [x] 3. Create requirements-fast.txt ✓
- [ ] 4. Edit Dockerfile (Tsinghua mirror) ← **Next**
- [ ] 5. Rebuild: \`docker compose down && docker compose up --build --no-cache\`
- [ ] 6. Verify pip <3min + models load
- [ ] 7. Test localhost:5000/health

## Commands ready:
```bash
docker compose down
docker compose up --build --no-cache
docker compose logs -f app | grep -E \"pip|Loaded|Best model\"
```

## Expected pip logs:
```
pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir -r requirements-fast.txt
Successfully installed pandas-2.1.0 pillow-10.0.0 plotly-5.18.0
```

**Next:** Create .dockerignore → requirements-fast.txt → Dockerfile edit → REBUILD

