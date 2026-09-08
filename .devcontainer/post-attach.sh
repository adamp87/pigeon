#!/usr/bin/env bash
# Post-attach lifecycle script for Pigeon DevContainer.
# Automatically initializes Coral Edge TPU hardware, triggers firmware
# upload/re-enumeration, and sets USB bus permissions.
set +e

export PATH="/home/user/.venv/bin:$PATH"
cd /workspaces/pigeon

if [ -d /dev/bus/usb ]; then
    echo "⚡ [DevContainer] Initializing Coral USB Edge TPU hardware..."

    MODEL="/workspaces/pigeon/data/models/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite"
    PYTHON_BIN="/home/user/.venv/bin/python3"
    [ -f "$PYTHON_BIN" ] || PYTHON_BIN="python3"

    if [ ! -f "$MODEL" ]; then
        echo "❌ [DevContainer] Error: Coral Edge TPU initialization model not found at $MODEL" >&2
        exit 1
    fi

    MAX_ATTEMPTS=5
    RETRY_SLEEP=10
    SUCCESS=0

    for attempt in $(seq 1 $MAX_ATTEMPTS); do
        echo "🔍 [DevContainer] TPU detection attempt $attempt/$MAX_ATTEMPTS..."

        # Grant permissions to any initial or newly created USB device nodes
        sudo chgrp -R plugdev /dev/bus/usb 2>/dev/null || true
        sudo chmod -R 0777 /dev/bus/usb 2>/dev/null || true

        # Run initialization & verification check in Python
        INIT_OUTPUT=$($PYTHON_BIN -c "
import sys
from pycoral.utils.edgetpu import list_edge_tpus, make_interpreter

tpus = list_edge_tpus()
if not tpus:
    print('NO_TPU')
    sys.exit(1)

print(f'DETECTED:{len(tpus)}:{tpus}')

ready = 0
for dev in [':0', ':1'][:len(tpus)]:
    try:
        interp = make_interpreter('$MODEL', device=dev)
        interp.allocate_tensors()
        ready += 1
        print(f'READY:{dev}')
    except Exception as e:
        print(f'ERROR:{dev}:{e}')

if ready == len(tpus) and ready > 0:
    sys.exit(0)
else:
    sys.exit(2)
" 2>/dev/null)
        RET=$?

        # Fix permissions again immediately after possible firmware re-enumeration
        sudo chgrp -R plugdev /dev/bus/usb 2>/dev/null || true
        sudo chmod -R 0777 /dev/bus/usb 2>/dev/null || true

        if [ $RET -eq 0 ]; then
            echo "✅ [DevContainer] Coral Edge TPU(s) initialized successfully on attempt $attempt:"
            echo "$INIT_OUTPUT" | grep "^DETECTED:" | sed 's/^DETECTED:/  - Devices: /'
            echo "$INIT_OUTPUT" | grep "^READY:" | sed 's/^READY:/  - Operational: /'
            SUCCESS=1
            break
        else
            if echo "$INIT_OUTPUT" | grep -q "^DETECTED:"; then
                echo "$INIT_OUTPUT" | grep "^DETECTED:" | sed 's/^DETECTED:/⚠️ [DevContainer] Devices detected: /'
                echo "$INIT_OUTPUT" | grep "^ERROR:" | sed 's/^ERROR:/  - Init warning: /'
            else
                echo "⚠️ [DevContainer] No Coral Edge TPU detected on attempt $attempt."
            fi

            if [ $attempt -lt $MAX_ATTEMPTS ]; then
                echo "⏳ [DevContainer] Waiting ${RETRY_SLEEP}s before retry..."
                sleep $RETRY_SLEEP
            fi
        fi
    done

    if [ $SUCCESS -eq 0 ]; then
        echo "⚠️ [DevContainer] Warning: Coral Edge TPU initialization could not verify operational devices after $MAX_ATTEMPTS attempts."
    fi
fi
exit 0
