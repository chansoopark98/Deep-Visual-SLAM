
const canvas = document.getElementById('renderCanvas');

// alpha 옵션 활성화 및 투명 배경 설정
const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x000000, 0); // 투명한 배경

// 씬 생성
const scene = new THREE.Scene();

// 원근 카메라(가상 카메라) 생성 및 초기 위치 설정
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 0, 10);

// AR 모델로 사용할 가상의 정육면체 생성 (1m x 1m x 1m)
const geometry = new THREE.BoxGeometry(1, 1, 1);
const material = new THREE.MeshBasicMaterial({ color: 0x00ff00, wireframe: true });
const arCube = new THREE.Mesh(geometry, material);

// 정육면체의 초기 위치 설정 (예: (1, 1, 1))
arCube.position.set(1, 1, 1);
scene.add(arCube);

function updateCameraPose(relativeMatrixArray) {
    // 1. tfjs tensor로부터 얻은 4x4 행렬 (2차원 배열에서 flat한 배열로 변환되어 column-major로 재배열된 상태라고 가정)
    const relMatrix = new THREE.Matrix4();
    relMatrix.fromArray(relativeMatrixArray);
    
    // 2. 필요에 따라 모델 좌표계(OpenCV 등)와 Three.js 좌표계의 차이 보정 (Y, Z 반전)
    // invert와 premultiply는 모델의 행렬 정의에 따라 적용
    relMatrix.invert();  // 모델의 행렬이 월드->카메라(view)라면, 역변환하여 카메라->월드로 만듦
    const cvToThree = new THREE.Matrix4().makeScale(1, -1, -1);
    relMatrix.premultiply(cvToThree);
    
    // 3. 현재 카메라의 절대 pose 행렬을 가져옵니다.
    const currentMatrix = camera.matrixWorld.clone();
    
    // 4. 상대 pose를 누적하여 새로운 절대 pose 계산
    // 새로운 카메라 pose = 현재 pose * 상대 변환
    const newMatrix = new THREE.Matrix4();
    newMatrix.multiplyMatrices(currentMatrix, relMatrix);
    
    // 5. 새 행렬을 분해하여 위치, 회전, 스케일을 추출
    const pos = new THREE.Vector3();
    const quat = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    newMatrix.decompose(pos, quat, scale);
    
    // 6. 스케일 보정 (필요한 경우, 여기서는 1:1 비율)
    pos.multiplyScalar(1.0);
    
    // 7. 카메라의 pose 업데이트
    camera.position.copy(pos);
    camera.quaternion.copy(quat);
    // 일반적으로 카메라 스케일은 (1,1,1)이어야 함
    camera.scale.set(1, 1, 1);
    
    // 8. 카메라 행렬 업데이트
    camera.updateMatrixWorld(true);
  }

function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
    console.log('Camera position:', camera.position);
    console.log('Camera rotation:', camera.rotation);
}

animate();
export { updateCameraPose };