'use strict';

const fs = require('fs');
const path = require('path');

function walkFiles(dir, baseDir = dir) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const files = [];

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);

    if (entry.isDirectory()) {
      files.push(...walkFiles(fullPath, baseDir));
      continue;
    }

    files.push({
      fullPath,
      relativePath: path.relative(baseDir, fullPath).split(path.sep).join('/')
    });
  }

  return files;
}

hexo.extend.generator.register('post-image-generator', function() {
  const imageRoot = path.join(hexo.source_dir, '_posts', 'image');

  if (!fs.existsSync(imageRoot)) {
    return [];
  }

  return walkFiles(imageRoot).map(file => ({
    path: `image/${file.relativePath}`,
    data: () => fs.createReadStream(file.fullPath)
  }));
});
