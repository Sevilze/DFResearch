use tch::{ Tensor, Kind };
use image::{ ImageBuffer, Luma };
use rustfft::{ FftPlanner, num_complex::Complex };
use rustdct::DctPlanner;
use ndarray::{ Array2, s };
use std::f32::consts::PI;
use super::model::InferenceError;

pub struct ChannelAugmentation {
    pub add_dwt: bool,
    pub add_dct: bool,
    pub add_sobel: bool,
    pub add_gray: bool,
    pub add_hog: bool,
    pub add_fft: bool,
    pub add_lbp: bool,
    pub add_ltp: bool,
    pub hog_orientations: usize,
    pub hog_pixels_per_cell: (usize, usize),
    pub lbp_n_points: usize,
    pub lbp_radius: usize,
    pub ltp_threshold: f32,
}

impl Default for ChannelAugmentation {
    fn default() -> Self {
        Self {
            add_dwt: true,
            add_dct: true,
            add_sobel: true,
            add_gray: false,
            add_hog: false,
            add_fft: true,
            add_lbp: true,
            add_ltp: false,
            hog_orientations: 9,
            hog_pixels_per_cell: (8, 8),
            lbp_n_points: 8,
            lbp_radius: 1,
            ltp_threshold: 0.1,
        }
    }
}

impl ChannelAugmentation {
    pub fn process_image(&self, image_bytes: &[u8]) -> Result<Tensor, InferenceError> {
        let img = image
            ::load_from_memory(image_bytes)
            .map_err(|e| InferenceError::PreprocessingError(e.to_string()))?
            .resize_exact(224, 224, image::imageops::FilterType::Triangle)
            .to_rgb8();

        let (width, height) = img.dimensions();

        let rgb_data: Vec<f32> = img
            .as_raw()
            .iter()
            .map(|v| (*v as f32) / 255.0)
            .collect();
        println!("Creating RGB tensor");
        let rgb_tensor = Tensor::f_from_slice(&rgb_data)
            .map_err(|e|
                InferenceError::PreprocessingError(format!("Failed to create RGB tensor: {:?}", e))
            )?
            .reshape(&[height as i64, width as i64, 3])
            .permute(&[2, 0, 1]);

        let mut channels: Vec<Tensor> = vec![rgb_tensor];

        let mut gray_img = ImageBuffer::<Luma<u8>, Vec<u8>>::new(width, height);
        for (x, y, pixel) in img.enumerate_pixels() {
            let r = pixel[0] as f32;
            let g = pixel[1] as f32;
            let b = pixel[2] as f32;
            let y_val = (0.2989 * r + 0.587 * g + 0.114 * b).round().clamp(0.0, 255.0) as u8;
            gray_img.put_pixel(x, y, Luma([y_val]));
        }

        let gray_np: Array2<f32> = Array2::from_shape_vec(
            (height as usize, width as usize),
            gray_img
                .as_raw()
                .iter()
                .map(|v| (*v as f32) / 255.0)
                .collect()
        ).map_err(|e|
            InferenceError::PreprocessingError(
                format!("Failed to create grayscale ndarray: {:?}", e)
            )
        )?;

        if self.add_gray {
            let gray_slice = gray_np
                .as_slice()
                .ok_or_else(||
                    InferenceError::PreprocessingError("Gray ndarray not contiguous".to_string())
                )?;
            let gray_tensor = Tensor::f_from_slice(gray_slice)
                .map_err(|e|
                    InferenceError::PreprocessingError(
                        format!("Failed to create grayscale tensor: {:?}", e)
                    )
                )?
                .reshape(&[height as i64, width as i64])
                .unsqueeze(0);
            channels.push(gray_tensor);
        }

        if self.add_fft {
            let fft_mag = fft_magnitude(&gray_np);
            channels.push(array_to_tensor(&fft_mag));
        }

        if self.add_dwt {
            let ll = haar_dwt_ll(&gray_np);

            let ll_norm = normalize_array(&ll);
            let (h_ll, w_ll) = ll_norm.dim();
            let mut ll_img = ImageBuffer::<Luma<u8>, Vec<u8>>::new(w_ll as u32, h_ll as u32);
            for y in 0..h_ll {
                for x in 0..w_ll {
                    let v = (ll_norm[[y, x]] * 255.0).clamp(0.0, 255.0) as u8;
                    ll_img.put_pixel(x as u32, y as u32, Luma([v]));
                }
            }

            // Resize with bilinear interpolation
            let resized_img = image::imageops::resize(
                &ll_img,
                width,
                height,
                image::imageops::FilterType::Triangle
            );

            // Convert back to Array2<f32>
            let mut ll_resized = Array2::<f32>::zeros((height as usize, width as usize));
            for (y, row) in resized_img.rows().enumerate() {
                for (x, pixel) in row.enumerate() {
                    ll_resized[[y, x]] = (pixel[0] as f32) / 255.0;
                }
            }

            channels.push(array_to_tensor(&ll_resized));
        }

        if self.add_dct {
            let dct_res = dct2d(&gray_np);
            channels.push(array_to_tensor(&dct_res));
        }

        if self.add_sobel {
            let sobel_res = sobel_edges(&gray_img);
            channels.push(array_to_tensor(&sobel_res));
        }

        if self.add_hog {
            let hog_res = hog_visualization(
                &gray_img,
                self.hog_orientations,
                self.hog_pixels_per_cell
            );
            channels.push(array_to_tensor(&hog_res));
        }

        if self.add_lbp {
            let lbp_res = lbp_uniform(&gray_img, self.lbp_n_points, self.lbp_radius);
            channels.push(array_to_tensor(&lbp_res));
        }

        if self.add_ltp {
            let ltp_res = ltp_pattern(&gray_np, self.ltp_threshold);
            channels.push(array_to_tensor(&ltp_res));
        }

        let fused = Tensor::cat(&channels, 0).unsqueeze(0);
        Ok(fused)
    }
}

fn normalize_array(arr: &Array2<f32>) -> Array2<f32> {
    let min = arr.fold(f32::INFINITY, |a, &b| a.min(b));
    let max = arr.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    arr.mapv(|x| (x - min) / (max - min + 1e-8))
}

fn array_to_tensor(arr: &Array2<f32>) -> Tensor {
    let arr_norm = normalize_array(arr);
    let slice = arr_norm.as_slice().expect("Normalized array not contiguous");
    Tensor::f_from_slice(slice)
        .expect("Failed to create tensor from normalized array")
        .reshape(&[arr.shape()[0] as i64, arr.shape()[1] as i64])
        .unsqueeze(0)
}

fn fft_magnitude(gray: &Array2<f32>) -> Array2<f32> {
    let (h, w) = gray.dim();
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(w);
    let mut data: Vec<Complex<f32>> = gray
        .outer_iter()
        .flat_map(|row|
            row
                .iter()
                .map(|&x| Complex { re: x, im: 0.0 })
                .collect::<Vec<_>>()
        )
        .collect();

    // 1D FFT on rows
    for y in 0..h {
        let start = y * w;
        let end = start + w;
        fft.process(&mut data[start..end]);
    }

    // transpose
    let mut transposed = vec![Complex{re:0.0, im:0.0}; h*w];
    for y in 0..h {
        for x in 0..w {
            transposed[x * h + y] = data[y * w + x];
        }
    }

    // FFT on columns
    let fft_col = planner.plan_fft_forward(h);
    for x in 0..w {
        let start = x * h;
        let end = start + h;
        fft_col.process(&mut transposed[start..end]);
    }

    // transpose back
    let mut final_data = vec![Complex{re:0.0, im:0.0}; h*w];
    for y in 0..h {
        for x in 0..w {
            final_data[y * w + x] = transposed[x * h + y];
        }
    }

    // fftshift + magnitude + log
    let mut mag = Array2::<f32>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let shifted_y = (y + h / 2) % h;
            let shifted_x = (x + w / 2) % w;
            let val = final_data[shifted_y * w + shifted_x];
            mag[[y, x]] = (val.norm() + 1e-8).ln();
        }
    }
    mag
}

fn haar_dwt_ll(gray: &Array2<f32>) -> Array2<f32> {
    // Biorthogonal 2.2 filters
    let dec_lo = vec![
        0.0,
        0.0,
        0.1767766952966369,
        0.3535533905932738,
        0.1767766952966369,
        0.0,
        0.0,
        0.0
    ];
    let (h, _w) = gray.dim();

    // Convolve rows
    let mut approx_rows = Vec::with_capacity(h);
    for row in gray.outer_iter() {
        let row_vec = row.to_vec();
        let approx = dwt_1d(&row_vec, &dec_lo);
        approx_rows.push(approx);
    }

    let half_w = approx_rows[0].len();
    let mut approx_matrix = Array2::<f32>::zeros((h, half_w));
    for (i, approx) in approx_rows.iter().enumerate() {
        for (j, &val) in approx.iter().enumerate() {
            approx_matrix[[i, j]] = val;
        }
    }

    // Determine output height dynamically based on first column's DWT output length
    let first_col: Vec<f32> = approx_matrix.column(0).to_vec();
    let approx_col0 = dwt_1d(&first_col, &dec_lo);
    let out_h = approx_col0.len();

    let mut ll = Array2::<f32>::zeros((out_h, half_w));
    for col_idx in 0..half_w {
        let col: Vec<f32> = approx_matrix.column(col_idx).to_vec();
        let approx_col = dwt_1d(&col, &dec_lo);
        let len = approx_col.len().min(out_h);
        for i in 0..len {
            ll[[i, col_idx]] = approx_col[i];
        }
    }

    ll
}

fn dwt_1d(signal: &[f32], low_filter: &[f32]) -> Vec<f32> {
    let filter_len = low_filter.len();
    let len = signal.len();
    let mut approx = Vec::with_capacity((len + 1) / 2);

    let pad = filter_len / 2;

    for i in (0..len + pad * 2).step_by(2) {
        let mut sum = 0.0;
        for k in 0..filter_len {
            let idx_signed = (i as isize) + (k as isize) - (pad as isize);
            let val = if idx_signed < 0 || idx_signed >= (len as isize) {
                0.0
            } else {
                signal[idx_signed as usize]
            };
            sum += low_filter[k] * val;
        }
        approx.push(sum);
    }
    approx
}

fn dct2d(gray: &Array2<f32>) -> Array2<f32> {
    let (h, w) = gray.dim();
    let mut planner = DctPlanner::new();
    let dct_h = planner.plan_dct2(h);
    let dct_w = planner.plan_dct2(w);

    let mut data = gray.to_owned();

    // DCT on rows
    for mut row in data.outer_iter_mut() {
        let row_slice = row.as_slice_mut().expect("DCT row not contiguous");
        dct_w.process_dct2(row_slice);
        // Orthonormal scaling for rows
        let scale = (2.0 * (w as f32).sqrt()).recip();
        for v in row_slice.iter_mut() {
            *v *= scale;
        }
    }

    // transpose
    let data_t = data.reversed_axes().to_owned();
    let mut data_t_fixed = data_t.to_owned();

    // DCT on columns (which are now rows in transposed view)
    for i in 0..data_t_fixed.len_of(ndarray::Axis(0)) {
        let mut col_vec = data_t_fixed.index_axis(ndarray::Axis(0), i).to_owned().into_raw_vec();
        dct_h.process_dct2(&mut col_vec);
        // Orthonormal scaling for columns
        let scale = (2.0 * (h as f32).sqrt()).recip();
        for v in col_vec.iter_mut() {
            *v *= scale;
        }
        let col_array = ndarray::Array1::from(col_vec);
        data_t_fixed.index_axis_mut(ndarray::Axis(0), i).assign(&col_array);
    }

    data_t.reversed_axes()
}

fn sobel_edges(gray_img: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Array2<f32> {
    let (w, h) = gray_img.dimensions();
    let w = w as usize;
    let h = h as usize;
    let mut grad_x = Array2::<f32>::zeros((h, w));
    let mut grad_y = Array2::<f32>::zeros((h, w));

    let kernel_x = [
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0],
    ];
    let kernel_y = [
        [-1.0, -2.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0],
    ];

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            for ky in 0..3 {
                for kx in 0..3 {
                    let pixel = gray_img.get_pixel((x + kx - 1) as u32, (y + ky - 1) as u32)
                        [0] as f32;
                    sum_x += kernel_x[ky][kx] * pixel;
                    sum_y += kernel_y[ky][kx] * pixel;
                }
            }
            grad_x[[y, x]] = sum_x;
            grad_y[[y, x]] = sum_y;
        }
    }

    let mut mag = Array2::<f32>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            mag[[y, x]] = (grad_x[[y, x]].powi(2) + grad_y[[y, x]].powi(2)).sqrt();
        }
    }
    mag
}

fn hog_visualization(
    gray_img: &ImageBuffer<Luma<u8>, Vec<u8>>,
    orientations: usize,
    pixels_per_cell: (usize, usize)
) -> Array2<f32> {
    let (w, h) = gray_img.dimensions();
    let w = w as usize;
    let h = h as usize;
    let mut grad_x = Array2::<f32>::zeros((h, w));
    let mut grad_y = Array2::<f32>::zeros((h, w));

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            grad_x[[y, x]] =
                (gray_img.get_pixel((x + 1) as u32, y as u32)[0] as f32) -
                (gray_img.get_pixel((x - 1) as u32, y as u32)[0] as f32);
            grad_y[[y, x]] =
                (gray_img.get_pixel(x as u32, (y + 1) as u32)[0] as f32) -
                (gray_img.get_pixel(x as u32, (y - 1) as u32)[0] as f32);
        }
    }

    let mut magnitude = Array2::<f32>::zeros((h, w));
    let mut orientation = Array2::<f32>::zeros((h, w));

    for y in 0..h {
        for x in 0..w {
            magnitude[[y, x]] = (grad_x[[y, x]].powi(2) + grad_y[[y, x]].powi(2)).sqrt();
            orientation[[y, x]] =
                ((grad_y[[y, x]].atan2(grad_x[[y, x]]) * 180.0) / PI + 180.0) % 180.0;
        }
    }

    let cell_h = pixels_per_cell.1;
    let cell_w = pixels_per_cell.0;
    let n_cells_y = h / cell_h;
    let n_cells_x = w / cell_w;

    let mut hog_image = Array2::<f32>::zeros((h, w));

    for cy in 0..n_cells_y {
        for cx in 0..n_cells_x {
            let mut hist = vec![0.0f32; orientations];
            for y in 0..cell_h {
                for x in 0..cell_w {
                    let yy = cy * cell_h + y;
                    let xx = cx * cell_w + x;
                    if yy >= h || xx >= w {
                        continue;
                    }
                    let bin =
                        (
                            (
                                (orientation[[yy, xx]] / 180.0) *
                                (orientations as f32)
                            ).floor() as usize
                        ) % orientations;
                    hist[bin] += magnitude[[yy, xx]];
                }
            }
            let max_hist = hist
                .iter()
                .cloned()
                .fold(0.0 / 0.0, f32::max);
            for y in 0..cell_h {
                for x in 0..cell_w {
                    let yy = cy * cell_h + y;
                    let xx = cx * cell_w + x;
                    if yy >= h || xx >= w {
                        continue;
                    }
                    hog_image[[yy, xx]] = max_hist;
                }
            }
        }
    }
    hog_image
}

fn lbp_uniform(
    gray_img: &ImageBuffer<Luma<u8>, Vec<u8>>,
    n_points: usize,
    radius: usize
) -> Array2<f32> {
    let (w, h) = gray_img.dimensions();
    let w = w as usize;
    let h = h as usize;
    let mut lbp = Array2::<f32>::zeros((h, w));

    for y in radius..h - radius {
        for x in radius..w - radius {
            let center = gray_img.get_pixel(x as u32, y as u32)[0] as f32;
            let mut pattern = Vec::with_capacity(n_points);
            for p in 0..n_points {
                let theta = (2.0 * PI * (p as f32)) / (n_points as f32);
                let dx = ((radius as f32) * theta.cos()).round() as isize;
                let dy = ((radius as f32) * theta.sin()).round() as isize;
                let neighbor = gray_img.get_pixel(
                    ((x as isize) + dx) as u32,
                    ((y as isize) + dy) as u32
                )[0] as f32;
                pattern.push(if neighbor >= center { 1 } else { 0 });
            }

            // Count transitions in circular pattern
            let mut transitions = 0;
            for i in 0..n_points {
                if pattern[i] != pattern[(i + 1) % n_points] {
                    transitions += 1;
                }
            }

            let uniform_code = if transitions <= 2 {
                pattern.iter().sum::<u32>() as f32
            } else {
                (n_points + 1) as f32 // non-uniform label
            };

            lbp[[y, x]] = uniform_code;
        }
    }

    // Normalize to [0,1]
    let max_val = (n_points + 1) as f32;
    lbp.mapv_inplace(|x| x / max_val);

    lbp
}

fn ltp_pattern(gray: &Array2<f32>, threshold: f32) -> Array2<f32> {
    let (h, w) = gray.dim();

    // Skip if too small
    if h < 3 || w < 3 {
        return Array2::<f32>::zeros((h, w));
    }

    let mut ltp = Array2::<f32>::zeros((h, w));

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let center = gray[[y, x]];
            let neighborhood = gray.slice(s![y - 1..=y + 1, x - 1..=x + 1]);
            let mut sum = 0.0;
            for v in neighborhood.iter() {
                let diff = *v - center;
                if diff > threshold {
                    sum += 1.0;
                } else if diff < -threshold {
                    sum -= 1.0;
                }
            }
            ltp[[y, x]] = sum;
        }
    }

    // Normalize to [0,1]
    let min = ltp.fold(f32::INFINITY, |a, &b| a.min(b));
    let max = ltp.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    if (max - min).abs() > 1e-8 {
        ltp.mapv_inplace(|x| (x - min) / (max - min + 1e-8));
    }

    ltp
}

pub fn preprocess(image_bytes: &[u8]) -> Result<Tensor, InferenceError> {
    let augmenter = ChannelAugmentation::default();
    let tensor = augmenter.process_image(image_bytes)?;

    let _num_channels = check_channels(&tensor)?;
    let tensor = tensor.to_kind(Kind::Float);

    Ok(tensor)
}

pub fn check_channels(tensor: &Tensor) -> Result<usize, InferenceError> {
    let shape = tensor.size();
    if shape.len() != 4 {
        return Err(
            InferenceError::PreprocessingError(
                format!(
                    "Expected tensor with 4 dimensions (batch, channels, height, width), got shape: {:?}",
                    shape
                )
            )
        );
    }

    if shape[2] != 224 || shape[3] != 224 {
        return Err(
            InferenceError::PreprocessingError(
                format!("Expected spatial size 224x224, got {}x{}", shape[2], shape[3])
            )
        );
    }

    let num_channels = shape[1] as usize;
    Ok(num_channels)
}
