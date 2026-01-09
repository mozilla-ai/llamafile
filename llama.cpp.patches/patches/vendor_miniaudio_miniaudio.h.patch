diff --git a/vendor/miniaudio/miniaudio.h b/vendor/miniaudio/miniaudio.h
--- a/llama.cpp/vendor/miniaudio/miniaudio.h
+++ b/llama.cpp/vendor/miniaudio/miniaudio.h
@@ -3858,7 +3858,7 @@ typedef ma_uint16 wchar_t;
 
 
 /* Platform/backend detection. */
-#if defined(_WIN32) || defined(__COSMOPOLITAN__)
+#if defined(_WIN32)
     #define MA_WIN32
     #if defined(MA_FORCE_UWP) || (defined(WINAPI_FAMILY) && ((defined(WINAPI_FAMILY_PC_APP) && WINAPI_FAMILY == WINAPI_FAMILY_PC_APP) || (defined(WINAPI_FAMILY_PHONE_APP) && WINAPI_FAMILY == WINAPI_FAMILY_PHONE_APP)))
         #define MA_WIN32_UWP
