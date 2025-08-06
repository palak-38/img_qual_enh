// server.js (example)
const express = require('express');
const multer = require('multer');
const cors = require('cors');

const app = express();
const upload = multer();

app.use(cors()); // If your frontend is on a different URL/port

app.post('/api/enhance', upload.single('image'), async (req, res) => {
  try {
    // 1) Get the uploaded file from req.file
    const originalImageBuffer = req.file.buffer;

    // 2) Run your "enhance" model (this is just pseudocode)
    // const { enhancedImageBuffer, psnr, ssim, mos } = await enhanceMarsImage(originalImageBuffer);

    // For demonstration, we’ll mock some data:
    const enhancedImageBuffer = originalImageBuffer; // pretend it's "enhanced" 
    const psnr = 32.47;
    const ssim = 0.95;
    const mos = 4.7;

    // 3) Store or serve the images somewhere accessible
    // For simplicity, let's assume we base64-encode them to embed directly.
    const inputBase64 = originalImageBuffer.toString('base64');
    const outputBase64 = enhancedImageBuffer.toString('base64');

    // 4) Return JSON with the base64 data (or you can save to a file and return a URL)
    res.json({
      inputImageURL: `data:image/png;base64,${inputBase64}`,
      outputImageURL: `data:image/png;base64,${outputBase64}`,
      psnr: psnr,
      ssim: ssim,
      mos: mos
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Enhancement failed.' });
  }
});

app.listen(3000, () => {
  console.log('Server running on http://localhost:3000');
});
