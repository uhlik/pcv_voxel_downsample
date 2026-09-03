use pyo3::prelude::*;
use pyo3::exceptions::{PyTypeError, PyValueError, PyRuntimeError};
use numpy::{PyArray1, PyReadonlyArray2, IntoPyArray, PyArrayMethods}; 
use rayon::slice::ParallelSliceMut;

#[derive(Copy, Clone)]
struct PointBin {
    voxel_key: u64,
    original_index: usize,
}

#[inline]
fn encode_voxel(
    x: f64, y: f64, z: f64,
    min_x: f64, min_y: f64, min_z: f64,
    voxel_size: f64,
) -> u64 {
    let ix = ((x - min_x) / voxel_size).floor() as i64 + 1_048_576;
    let iy = ((y - min_y) / voxel_size).floor() as i64 + 1_048_576;
    let iz = ((z - min_z) / voxel_size).floor() as i64 + 1_048_576;

    (((ix & 0x1F_FFFF) as u64) << 42) |
    (((iy & 0x1F_FFFF) as u64) << 21) |
    ((iz & 0x1F_FFFF) as u64)
}

fn voxel_downsample_core(
    points: &[f64],
    num_points: usize,
    voxel_size: f64,
) -> Result<(Vec<f64>, Vec<i64>), String> {
    if num_points == 0 || voxel_size <= 0.0 {
        return Ok((Vec::new(), Vec::new()));
    }

    // Compute point cloud bounding box
    let (mut min_x, mut max_x) = (points[0], points[0]);
    let (mut min_y, mut max_y) = (points[1], points[1]);
    let (mut min_z, mut max_z) = (points[2], points[2]);

    for i in 1..num_points {
        let x = points[i * 3];
        let y = points[i * 3 + 1];
        let z = points[i * 3 + 2];
        if x < min_x { min_x = x; } else if x > max_x { max_x = x; }
        if y < min_y { min_y = y; } else if y > max_y { max_y = y; }
        if z < min_z { min_z = z; } else if z > max_z { max_z = z; }
    }

    let span_x = (max_x - min_x) / voxel_size;
    let span_y = (max_y - min_y) / voxel_size;
    let span_z = (max_z - min_z) / voxel_size;

    if span_x >= 1_048_575.0 || span_y >= 1_048_575.0 || span_z >= 1_048_575.0 {
        return Err(format!(
            "Point cloud bounding box dimensions ({}x, {}y, {}z grid units) \
             exceed the max 21-bit limit (1,048,575 units) for the current voxel size: {}",
            span_x, span_y, span_z, voxel_size
        ));
    }

    // Allocate spatial grid bins
    let mut bins = Vec::with_capacity(num_points);
    for i in 0..num_points {
        let key = encode_voxel(
            points[i * 3], points[i * 3 + 1], points[i * 3 + 2],
            min_x, min_y, min_z, voxel_size
        );
        bins.push(PointBin { voxel_key: key, original_index: i });
    }

    // Multi-threaded unstable sort via Rayon
    bins.par_sort_unstable_by_key(|bin| bin.voxel_key);

    let mut out_points = Vec::new();
    let mut out_indices = Vec::new();

    let mut start = 0;
    while start < num_points {
        let mut end = start;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_z = 0.0;

        while end < num_points && bins[end].voxel_key == bins[start].voxel_key {
            let idx = bins[end].original_index;
            sum_x += points[idx * 3];
            sum_y += points[idx * 3 + 1];
            sum_z += points[idx * 3 + 2];
            end += 1;
        }

        let count = (end - start) as f64;
        let avg_x = sum_x / count;
        let avg_y = sum_y / count;
        let avg_z = sum_z / count;

        let mut min_dist_sq = f64::MAX;
        let mut chosen_idx = 0;

        for i in start..end {
            let idx = bins[i].original_index;
            let dx = points[idx * 3] - avg_x;
            let dy = points[idx * 3 + 1] - avg_y;
            let dz = points[idx * 3 + 2] - avg_z;
            let dist_sq = dx * dx + dy * dy + dz * dz;

            if dist_sq < min_dist_sq {
                min_dist_sq = dist_sq;
                chosen_idx = idx;
            }
        }

        out_points.push(points[chosen_idx * 3]);
        out_points.push(points[chosen_idx * 3 + 1]);
        out_points.push(points[chosen_idx * 3 + 2]);
        out_indices.push(chosen_idx as i64);

        start = end;
    }

    Ok((out_points, out_indices))
}

#[pyfunction]
#[pyo3(name = "voxel_downsample")]
fn py_voxel_downsample<'py>(
    py: Python<'py>,
    points_obj: &Bound<'py, PyAny>,
    voxel_size: f64,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyArray1<i64>>)> {
    // 1. Fallback safe type normalization to a numpy array instance
    let np_module = py.import_bound("numpy")?;
    let points_arr = if points_obj.is_instance_of::<numpy::PyArray2<f64>>() 
                     || points_obj.is_instance_of::<numpy::PyArray2<f32>>() {
        points_obj.clone()
    } else {
        np_module.call_method1("asarray", (points_obj,))?
    };

    // Extract type context
    let dtype = points_arr.getattr("dtype")?;
    let dtype_name: String = dtype.getattr("name")?.extract()?;

    let is_f32 = match dtype_name.as_str() {
        "float32" => true,
        "float64" => false,
        _ => return Err(PyTypeError::new_err(format!("Must be np.float32 or np.float64. Got: {}", dtype_name))),
    };

    // 2. Structural constraint tracking
    let ndim: usize = points_arr.getattr("ndim")?.extract()?;
    let shape: Vec<usize> = points_arr.getattr("shape")?.extract()?;
    if ndim != 2 || shape[1] != 3 {
        return Err(PyValueError::new_err(format!("Input point cloud must have shape (N, 3). Got: {:?}", shape)));
    }

    // 3. Finite value assertions (NaN / Inf)
    let is_finite: bool = np_module
        .call_method1("all", (np_module.call_method1("isfinite", (&points_arr,))?,))?
        .extract()?;
    if !is_finite {
        return Err(PyValueError::new_err("Input point cloud contains NaN or Inf values."));
    }

    // 4. Thread-safe data normalization
    let num_points = shape[0];
    let mut points_f64_vec = Vec::with_capacity(num_points * 3);

    if is_f32 {
        let readonly_arr: PyReadonlyArray2<f32> = points_arr.extract()?;
        let view = readonly_arr.as_array();
        for row in view.rows() {
            points_f64_vec.extend_from_slice(&[row[0] as f64, row[1] as f64, row[2] as f64]);
        }
    } else {
        let readonly_arr: PyReadonlyArray2<f64> = points_arr.extract()?;
        let view = readonly_arr.as_array();
        for row in view.rows() {
            points_f64_vec.extend_from_slice(&[row[0], row[1], row[2]]);
        }
    };

    // 5. Drop the Python GIL to compute operations in absolute parallel
    let (out_points, out_indices) = py.allow_threads(|| {
        voxel_downsample_core(&points_f64_vec, num_points, voxel_size)
    }).map_err(|e| PyRuntimeError::new_err(e))?;

    let out_num_points = out_indices.len();
    let res_indices_arr = out_indices.into_pyarray_bound(py);

    // Reconstruct output shapes maintaining input precision constraints
    if is_f32 {
        let out_points_f32: Vec<f32> = out_points.into_iter().map(|p| p as f32).collect();
        let flat_arr = out_points_f32.into_pyarray_bound(py);
        let reshaped = flat_arr.reshape([out_num_points, 3])?;
        Ok((reshaped.into_any(), res_indices_arr))
    } else {
        let flat_arr = out_points.into_pyarray_bound(py);
        let reshaped = flat_arr.reshape([out_num_points, 3])?;
        Ok((reshaped.into_any(), res_indices_arr))
    }
}

#[pymodule]
fn pcv_voxel_downsample(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_voxel_downsample, m)?)?;
    Ok(())
}
