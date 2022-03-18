import numpy as np
from scipy.io import loadmat, savemat
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    blending
)

class FaceModel():
    def __init__(self, model_path, delta_bs_path, device=torch.device('cuda')):
        model = loadmat(model_path)
        # mean face shape. [1, N*3]
        self.meanshape = torch.from_numpy(model['meanshape'])
        # identity basis. [N*3,80]
        self.idBase = torch.from_numpy(model['idBase'])
        # mean face texture. [1, N*3] (0-255)
        self.meantex = torch.from_numpy(model['meantex'])
        # texture basis. [N*3,80]
        self.texBase = torch.from_numpy(model['texBase'])
        # triangle indices for each vertex that lies in. starts from 1. [N,8]
        self.point_buf = torch.from_numpy(model['point_buf']).long()
        # vertex indices in each triangle. starts from 1. [F,3]
        self.tri = torch.from_numpy(model['tri']).long()
        # vertex indices of 68 facial landmarks. starts from 0. [68]
        self.kp_inds = torch.tensor(model['keypoints'] - 1).squeeze().long()
        # delta blendshapes. [N*3, 46]
        delta_bs = np.load(delta_bs_path)
        self.deltaB = torch.from_numpy(delta_bs).float() # to float32

        self.n_ver = int(self.idBase.size(0) / 3)
        self.n_id_para = self.idBase.size(1)
        self.n_bs_para = self.deltaB.size(1)
        self.n_tex_para = self.texBase.size(1)
        # predefine
        self.n_rot_para = 3
        self.n_light_para = 27
        self.n_tran_para = 3

        # print('3DMM parameter')
        # print('n_id_para: ', self.n_id_para)
        # print('n_bs_para: ', self.n_bs_para)
        # print('n_tex_para: ', self.n_tex_para)
        # print('n_rot_para: ', self.n_rot_para)
        # print('n_light_para: ', self.n_light_para)
        # print('n_tran_para: ', self.n_tran_para)

        # init device
        self.device = device
        self.to_device(device)

    def get_neutral_face(self, id_coeff):
        '''
        get neutral face shape
        '''
        # [bs, nverx3]
        neutral_face = self.meanshape + torch.einsum('ij,aj->ai', self.idBase, id_coeff)
        return neutral_face

    def get_blendshapes(self, id_coeff, add_neutral_face=False):
        neutral_face = self.get_neutral_face(id_coeff)

        blendshapes = []
        if add_neutral_face:
            blendshapes.append(neutral_face)

        for bs_idx in range(self.n_bs_para):
            bs_coeff = torch.zeros(1, self.n_bs_para)
            bs_coeff[0, bs_idx] = 1.0

            B = neutral_face + torch.einsum('ij,aj->ai', self.deltaB, bs_coeff)
            blendshapes.append(B)

        return blendshapes


    def to_device(self, device):
        self.meanshape = self.meanshape.to(device)
        self.idBase = self.idBase.to(device)
        self.meantex = self.meantex.to(device)
        self.texBase = self.texBase.to(device)
        self.deltaB = self.deltaB.to(device)
        self.point_buf = self.point_buf.to(device)
        self.tri = self.tri.to(device)
        self.kp_inds =self.kp_inds.to(device)



class FaceDecoder():
    def __init__(self, model_path, delta_bs_path, device, batch_size,
                 focal=1015, img_size=224):

        facemodel = FaceModel(model_path, delta_bs_path, device)
        self.facemodel = facemodel

        self.focal = focal
        self.img_size = img_size
        # self.img_size = 300
        self.n_b = batch_size

        self.device = device

        self.renderer = self.get_renderer(self.device)
        self.renderer.to(self.device)

        self.build_constant()

    def build_constant(self):

        # SH illumination
        a0 = np.pi
        a1 = 2 * np.pi / np.sqrt(3.0)
        a2 = 2 * np.pi / np.sqrt(8.0)
        c0 = 1 / np.sqrt(4 * np.pi)
        c1 = np.sqrt(3.0) / np.sqrt(4 * np.pi)
        c2 = 3 * np.sqrt(5.0) / np.sqrt(12 * np.pi)
        d0 = 0.5 / np.sqrt(3.0)

        self.a0 = torch.tensor(a0, dtype=torch.float32, device=self.device)
        self.a1 = torch.tensor(a1, dtype=torch.float32, device=self.device)
        self.a2 = torch.tensor(a2, dtype=torch.float32, device=self.device)
        self.c0 = torch.tensor(c0, dtype=torch.float32, device=self.device)
        self.c1 = torch.tensor(c1, dtype=torch.float32, device=self.device)
        self.c2 = torch.tensor(c2, dtype=torch.float32, device=self.device)
        self.d0 = torch.tensor(d0, dtype=torch.float32, device=self.device)

        num_vertex = self.facemodel.n_ver
        n_v_full = self.n_b * num_vertex

        Y0 = torch.ones(n_v_full).to(self.device).float() * self.a0 * self.c0
        self.Y0 = Y0

        # Projection
        half_image_width = self.img_size // 2

        camera_pos = torch.tensor([0.0, 0.0, 10.0], device=self.device).reshape(1, 1, 3)

        p_matrix = torch.tensor([self.focal, 0.0, half_image_width,
                             0.0, self.focal, half_image_width,
                             0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)

        self.camera_pos = camera_pos
        self.p_matrix = p_matrix


    def get_renderer(self, device):
        R, T = look_at_view_transform(10, 0, 0)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.01, zfar=50,
                                        fov=2 * np.arctan(self.img_size // 2 / self.focal) * 180. / np.pi)

        lights = PointLights(device=device, location=[[0.0, 0.0, 1e5]], ambient_color=[[1, 1, 1]],
                             specular_color=[[0., 0., 0.]], diffuse_color=[[0., 0., 0.]])

        raster_settings = RasterizationSettings(
            image_size=self.img_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        blend_params = blending.BlendParams(background_color=[0, 0, 0])

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=cameras,
                lights=lights,
                blend_params=blend_params
            )
        )
        return renderer


    def Split_coeff(self, coeff):
        to_id = self.facemodel.n_id_para
        to_bs = to_id + self.facemodel.n_bs_para
        to_tex = to_bs + self.facemodel.n_tex_para
        to_angle = to_tex + self.facemodel.n_rot_para
        to_gamma = to_angle + self.facemodel.n_light_para
        to_trans = to_gamma + self.facemodel.n_tran_para

        id_coeff = coeff[:, :to_id]
        bs_coeff = coeff[:, to_id:to_bs]  # blendshape coeff
        tex_coeff = coeff[:, to_bs:to_tex]  # texture(albedo) coeff
        angles = coeff[:, to_tex:to_angle]  # ruler angles(x,y,z) for rotation
        gamma = coeff[:, to_angle:to_gamma]  # lighting coeff for 3 channel SH function
        translation = coeff[:, to_gamma:to_trans]  # translation coeff


        return id_coeff, bs_coeff, tex_coeff, angles, gamma, translation


    def Shape_formation(self, id_coeff, bs_coeff):
        # compute face shape with blendshape coeff, based on BFM model
        # input: bs_coeff with shape [bs,46], bs is batch size
        #        id_coeff with shape [bs,80]
        # output: face_shape with shape [bs,N,3], N is number of vertices

        '''
            S = neutral_shape + \alpha * deltaB
        '''
        n_b = bs_coeff.size(0)

        B0 = self.facemodel.get_neutral_face(id_coeff) # [bs, nverx3]

        # [bs, nverx3]
        face_shape = B0 + torch.einsum('ij,aj->ai', self.facemodel.deltaB, bs_coeff)
        face_shape = face_shape.view(n_b, -1, 3)

        return face_shape


    def Texture_formation(self, tex_coeff):
        # compute vertex texture(albedo) with tex_coeff
        # input: tex_coeff with shape [bs,N,3]
        # output: face_texture with shape [bs,N,3], RGB order, range from 0-255

        '''
            T = mean_texture + \lambda * B_texture
        '''
        n_b = tex_coeff.size(0)
        # [bs, nverx3]
        face_texture = self.facemodel.meantex + torch.einsum('ij,aj->ai', self.facemodel.texBase, tex_coeff)
        face_texture = face_texture.view(n_b, -1, 3)

        return face_texture


    def Compute_norm(self, face_shape):
        # compute vertex normal using one-ring neighborhood (8 points)
        # input: face_shape with shape [bs,N,3]
        # output: v_norm with shape [bs,N,3]

        face_id = self.facemodel.tri - 1
        point_id = self.facemodel.point_buf - 1
        shape = face_shape
        v1 = shape[:, face_id[:, 0], :]
        v2 = shape[:, face_id[:, 1], :]
        v3 = shape[:, face_id[:, 2], :]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = e1.cross(e2)
        empty = torch.zeros((face_norm.size(0), 1, 3), dtype=face_norm.dtype, device=face_norm.device)
        face_norm = torch.cat((face_norm, empty), 1)
        v_norm = face_norm[:, point_id, :].sum(2)
        v_norm = v_norm / v_norm.norm(dim=2).unsqueeze(2)

        return v_norm


    def Projection_block(self, face_shape):
        # we choose the focal length and camera position empirically
        # project 3D face onto image plane
        # input: face_shape with shape [bs,N,3]
        # output: face_projection with shape [bs,N,2]
        batchsize = face_shape.shape[0]

        camera_pos = self.camera_pos

        p_matrix = self.p_matrix.view(1,3,3).repeat(batchsize, 1, 1)

        reverse_z = torch.tensor([1.0, 0, 0, 0, 1, 0, 0, 0, -1.0], dtype=torch.float32, device=face_shape.device)
        reverse_z = reverse_z.view(1, 3, 3).repeat(batchsize, 1, 1)


        face_shape = torch.matmul(face_shape, reverse_z) + camera_pos
        aug_projection = torch.matmul(face_shape, p_matrix.permute((0, 2, 1)))

        face_projection = aug_projection[:, :, :2] / \
                          torch.reshape(aug_projection[:, :, 2], [batchsize, -1, 1])
        return face_projection


    def Projection_block_3d(self, face_shape):
        # we choose the focal length and camera position empirically
        # project 3D face onto image plane
        # input: face_shape with shape [bs,N,3]
        # output: face_projection with shape [bs,N,2]
        batchsize = face_shape.shape[0]

        camera_pos = self.camera_pos

        p_matrix = self.p_matrix.view(1,3,3).repeat(batchsize, 1, 1)

        reverse_z = torch.tensor([1.0, 0, 0, 0, 1, 0, 0, 0, -1.0], dtype=torch.float32, device=face_shape.device)
        reverse_z = reverse_z.view(1, 3, 3).repeat(batchsize, 1, 1)


        face_shape = torch.matmul(face_shape, reverse_z) + camera_pos
        aug_projection = torch.matmul(face_shape, p_matrix.permute((0, 2, 1)))

        # face_projection = aug_projection[:, :, :2] / \
        #                   torch.reshape(aug_projection[:, :, 2], [batchsize, -1, 1])

        aug_projection[:, :, :2] = aug_projection[:, :, :2] / \
                          torch.reshape(aug_projection[:, :, 2], [batchsize, -1, 1])

        face_projection = aug_projection

        face_projection[:, :, 2] =  face_projection[:, :, 2] - face_projection[:, :, 2].min()
        z_scale = 160 / face_projection[:, :, 2].max()
        face_projection[:, :, 2] = face_projection[:, :, 2] * z_scale

        return face_projection

    def Illumination_layer(self, face_texture, norm, gamma):
        # compute vertex color using face_texture and SH function lighting approximation
        # input: face_texture with shape [1,N,3]
        #          norm with shape [1,N,3]
        #         gamma with shape [1,27]
        # output: face_color with shape [1,N,3], RGB order, range from 0-255

        n_b, num_vertex, _ = face_texture.size()
        n_v_full = n_b * num_vertex
        gamma = gamma.view(-1, 3, 9).clone()
        gamma[:, :, 0] += 0.8

        gamma = gamma.permute(0, 2, 1)

        # Y0 = torch.ones(n_v_full).to(gamma.device).float() * self.a0 * self.c0
        norm = norm.view(-1, 3)
        nx, ny, nz = norm[:, 0], norm[:, 1], norm[:, 2]
        arrH = []

        arrH.append(self.Y0)
        arrH.append(-self.a1 * self.c1 * ny)
        arrH.append(self.a1 * self.c1 * nz)
        arrH.append(-self.a1 * self.c1 * nx)
        arrH.append(self.a2 * self.c2 * nx * ny)
        arrH.append(-self.a2 * self.c2 * ny * nz)
        arrH.append(self.a2 * self.c2 * self.d0 * (3 * nz.pow(2) - 1))
        arrH.append(-self.a2 * self.c2 * nx * nz)
        arrH.append(self.a2 * self.c2 * 0.5 * (nx.pow(2) - ny.pow(2)))

        H = torch.stack(arrH, 1)
        Y = H.view(n_b, num_vertex, 9)
        lighting = Y.bmm(gamma)

        face_color = face_texture * lighting
        return face_color


    def Compute_rotation_matrix(self, angles):
        n_b = angles.size(0)
        sinx = torch.sin(angles[:, 0])
        siny = torch.sin(angles[:, 1])
        sinz = torch.sin(angles[:, 2])
        cosx = torch.cos(angles[:, 0])
        cosy = torch.cos(angles[:, 1])
        cosz = torch.cos(angles[:, 2])

        rotXYZ = torch.eye(3).view(1, 3, 3).repeat(n_b * 3, 1, 1).view(3, n_b, 3, 3)

        if angles.is_cuda: rotXYZ = rotXYZ.cuda()

        rotXYZ[0, :, 1, 1] = cosx
        rotXYZ[0, :, 1, 2] = -sinx
        rotXYZ[0, :, 2, 1] = sinx
        rotXYZ[0, :, 2, 2] = cosx
        rotXYZ[1, :, 0, 0] = cosy
        rotXYZ[1, :, 0, 2] = siny
        rotXYZ[1, :, 2, 0] = -siny
        rotXYZ[1, :, 2, 2] = cosy
        rotXYZ[2, :, 0, 0] = cosz
        rotXYZ[2, :, 0, 1] = -sinz
        rotXYZ[2, :, 1, 0] = sinz
        rotXYZ[2, :, 1, 1] = cosz

        rotation = rotXYZ[2].bmm(rotXYZ[1]).bmm(rotXYZ[0])

        return rotation.permute(0, 2, 1)


    def Rigid_transform_block(self, face_shape, rotation, translation):
        face_shape_r = torch.matmul(face_shape, rotation)
        face_shape_t = face_shape_r + translation.view(-1, 1, 3)

        return face_shape_t


    def get_landmarks(self, face_shape, kp_inds):
        lms = face_shape[:, kp_inds, :]
        return lms


    def decode_face(self, coeff, return_coeffs=False, return_neutral_face=False):


        id_coeff, bs_coeff, tex_coeff, angles, gamma, translation = self.Split_coeff(coeff)


        batch_num = bs_coeff.shape[0]

        face_shape = self.Shape_formation(id_coeff, bs_coeff)
        face_texture = self.Texture_formation(tex_coeff)
        face_norm = self.Compute_norm(face_shape)
        rotation = self.Compute_rotation_matrix(angles)
        face_norm_r = face_norm.bmm(rotation)
        face_shape_t = self.Rigid_transform_block(face_shape, rotation, translation)
        face_color = self.Illumination_layer(face_texture, face_norm_r, gamma)
        landmarks_3d = self.get_landmarks(face_shape_t, self.facemodel.kp_inds)
        landmarks_2d = self.Projection_block(landmarks_3d)
        landmarks_2d = torch.stack([landmarks_2d[:, :, 0], self.img_size - 1.0 - landmarks_2d[:, :, 1]], dim=2)

        if return_neutral_face:
            face_shape_neutral = self.facemodel.get_neutral_face(id_coeff)
            face_shape_neutral = face_shape_neutral.view(1, -1, 3)
            face_shape_neutral_t = self.Rigid_transform_block(face_shape_neutral, rotation, translation)
            landmarks_3d_neutral = self.get_landmarks(face_shape_neutral_t, self.facemodel.kp_inds)
            landmarks_2d_neutral = self.Projection_block(landmarks_3d_neutral)
            landmarks_2d_neutral = torch.stack([landmarks_2d_neutral[:, :, 0], self.img_size - 1.0 - landmarks_2d_neutral[:, :, 1]], dim=2)

        tri = self.facemodel.tri - 1
        face_color_pytorch3d = TexturesVertex(face_color)
        mesh = Meshes(face_shape_t, tri.repeat(batch_num, 1, 1), face_color_pytorch3d)
        rendered_img = self.renderer(mesh)
        rendered_img = torch.clamp(rendered_img, 0, 255)


        if return_coeffs:
            coeffs = {
                'coeff_id': id_coeff,
                'coeff_bs': bs_coeff,
                'coeff_angle': angles,
                'coeff_translation': translation,
                'coeff_tex': tex_coeff,
                'coeff_light': gamma
            }

            mesh_tuple = (face_shape_t, tri.repeat(batch_num, 1, 1), torch.clamp(face_color, 0, 255))

            if not return_neutral_face:
                return rendered_img, landmarks_2d, coeffs, mesh_tuple
            else:
                neutral_tuple = (face_shape_neutral, face_shape_neutral_t, landmarks_2d_neutral)
                return rendered_img, landmarks_2d, coeffs, mesh_tuple, neutral_tuple

        else:
            return rendered_img, landmarks_2d, bs_coeff, id_coeff, tex_coeff


    def get_face_on_2d_plane(self, coeff):

        id_coeff, bs_coeff, tex_coeff, angles, gamma, translation = self.Split_coeff(coeff)

        batch_num = bs_coeff.shape[0]

        face_shape = self.Shape_formation(id_coeff, bs_coeff)
        face_texture = self.Texture_formation(tex_coeff)
        face_norm = self.Compute_norm(face_shape)
        rotation = self.Compute_rotation_matrix(angles)
        face_norm_r = face_norm.bmm(rotation)
        face_shape_t = self.Rigid_transform_block(face_shape, rotation, translation)
        # face_color = self.Illumination_layer(face_texture, face_norm_r, gamma)
        # landmarks_3d = self.get_landmarks(face_shape_t, self.facemodel.kp_inds)
        # landmarks_2d = self.Projection_block(landmarks_3d)
        # landmarks_2d = torch.stack([landmarks_2d[:, :, 0], self.img_size - 1.0 - landmarks_2d[:, :, 1]], dim=2)

        face_shape_2d = self.Projection_block_3d(face_shape_t)

        face_shape_2d[:, :, 1] = self.img_size - 1.0 - face_shape_2d[:, :, 1]

        return face_shape_2d

