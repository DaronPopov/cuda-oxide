/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::{stream_pool, teardown_lock, Device, Stream};
use crate::{
    cudarc_shim::{ctx, primary_ctx, stream},
    error::DriverError,
};

impl Device {
    /// Explicitly releases this device's retained primary context.
    ///
    /// Call this after all derived streams, modules, events, and other handles
    /// have been closed and dropped. On failure the device remains armed, so
    /// the caller can report the error and retry. `Drop` makes a best-effort
    /// attempt if the device is later abandoned.
    pub fn close(&mut self) -> Result<(), DriverError> {
        if !self.owned || self.cu_ctx.is_null() {
            return Ok(());
        }
        let _guard = teardown_lock();
        self.bind_to_thread()?;
        if ctx::get_current()? == Some(self.cu_ctx) {
            unsafe { ctx::set_current(std::ptr::null_mut()) }?;
        }
        // Keep the pool out of circulation until the retain is released. If
        // release fails, restore its handles while the context is still owned.
        let mut pool = stream_pool()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let idle = pool.remove(&self.ordinal).unwrap_or_default();
        if let Err(error) = unsafe { primary_ctx::release(self.cu_device) } {
            pool.entry(self.ordinal).or_default().extend(idle);
            return Err(error);
        }
        drop(pool);
        self.cu_ctx = std::ptr::null_mut();
        self.owned = false;
        Ok(())
    }

    pub(super) fn drop_close(&mut self) {
        let _guard = teardown_lock();
        stream_pool()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&self.ordinal);
        let _ = self.bind_to_thread();
        let ctx = std::mem::replace(&mut self.cu_ctx, std::ptr::null_mut());
        if !ctx.is_null() {
            let _ = unsafe { primary_ctx::release(self.cu_device) };
        }
    }
}

impl Stream {
    /// Synchronizes and explicitly destroys this stream.
    ///
    /// Unlike `Drop`, this reports driver failures and leaves the stream armed
    /// when cleanup fails, so the caller can retry. Explicit close destroys
    /// the handle even on an owned device; ordinary drop may return it to the
    /// crate's stream pool.
    pub fn close(&mut self) -> Result<(), DriverError> {
        if !self.owned || self.cu_stream.is_null() {
            return Ok(());
        }
        self.device.bind_to_thread()?;
        let _guard = teardown_lock();
        unsafe { stream::synchronize(self.cu_stream) }?;
        unsafe { stream::destroy(self.cu_stream) }?;
        self.cu_stream = std::ptr::null_mut();
        self.owned = false;
        Ok(())
    }
}
