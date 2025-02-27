// main.js

// worker 생성 (worker 파일 경로에 맞게 수정)
const inferenceWorker = new Worker('../modules/tf_raw.js');
console.log('Inference worker created:', inferenceWorker);

// canvas 엘리먼트 (카메라 영상이 그려진 요소)
const canvas = document.getElementById('canvas');
const inferenceInterval = 100; // ms 단위 간격

// inference 결과를 처리할 콜백 함수
function handleResult(result) {
  // result는 worker에서 전달한 예측 결과 (예: 배열)
  console.log('Inference result received:', result);
  // UI 업데이트나 추가 처리 로직 구현 가능
}

// worker로부터 메시지 수신
inferenceWorker.onmessage = (event) => {
  const msg = event.data;
  if (msg.type === 'status') {
    console.log('Worker status:', msg.message);
  } else if (msg.type === 'result') {
    handleResult(msg.result);
  } else if (msg.type === 'error') {
    console.error('Worker error:', msg.message);
  }
};

// canvas에서 프레임 캡처하여 worker로 전송
function sendFrameToWorker() {
  // OffscreenCanvas 사용 고려 가능 (지원 브라우저에서)
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  const imageData = ctx.getImageData(0, 0, width, height);

  // worker에 전달: transferable object로 buffer 전송
  inferenceWorker.postMessage({
    type: 'frame',
    data: {
      buffer: imageData.data.buffer,
      width,
      height
    }
  }, [imageData.data.buffer]);
}

// 일정 간격으로 프레임 전송 (추론만 worker로 위임)
function startContinuousInference() {
  function inferLoop() {
    sendFrameToWorker();
    setTimeout(inferLoop, inferenceInterval);
  }
  inferLoop();
}

// 추론 시작
startContinuousInference();
