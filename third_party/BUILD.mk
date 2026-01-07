include third_party/double-conversion/BUILD.mk
include third_party/mbedtls/BUILD.mk
include third_party/sqlite/BUILD.mk
include third_party/stb/BUILD.mk

.PHONY: o/$(MODE)/third_party
o/$(MODE)/third_party: \
		o/$(MODE)/third_party/double-conversion \
		o/$(MODE)/third_party/mbedtls \
		o/$(MODE)/third_party/sqlite \
		o/$(MODE)/third_party/stb \
		o/$(MODE)/third_party/zipalign

# ==============================================================================
# zipalign
# ==============================================================================

PKGS += ZIPALIGN

o/$(MODE)/third_party/zipalign/zipalign: o/$(MODE)/third_party/zipalign/zipalign.o
	@mkdir -p $(@D)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

o/$(MODE)/third_party/zipalign/%.o: third_party/zipalign/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -I$(COSMOCC)/include/third_party/zlib -c -o $@ $<

.PHONY: o/$(MODE)/third_party/zipalign
o/$(MODE)/third_party/zipalign: o/$(MODE)/third_party/zipalign/zipalign

