const canvas = document.getElementById('renderCanvas');

// alpha 옵션 활성화 및 투명 배경 설정
const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x000000, 0); // 투명한 배경

// 씬 생성
const scene = new THREE.Scene();

// 카메라 생성 (원근 카메라)
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 10;

// BoxGeometry와 MeshBasicMaterial을 이용해 상자(AR Box) 생성
const geometry = new THREE.BoxGeometry(1, 1, 1);
const material = new THREE.MeshBasicMaterial({ color: 0x00ff00, wireframe: true });
const arBox = new THREE.Mesh(geometry, material);

// 상자의 위치를 (1, 1, 1)로 설정
arBox.position.set(1, 1, 1);
scene.add(arBox);

// 애니메이션 루프 함수 (렌더링 반복)
function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
    console.log('animate!');
}

animate();
