import re
import logging
import json
import pandas as pd
from typing import Any, Dict, List, Optional, Union
import fsspec
import boto3
from botocore.exceptions import ClientError


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StorageClient:
    """
    Cliente de armazenamento para S3, GCS e sistema de arquivos local utilizando fsspec.
    """

    def __init__(
        self,
        gcp_credentials: Optional[Union[str, Dict[str, Any]]] = None,
        aws_key_id: Optional[str] = None,
        aws_secret: Optional[str] = None,
        aws_region: Optional[str] = None,
    ):
        """
        Inicializa o StorageClient com credenciais opcionais para S3 e GCS.

        Args:
            gcp_credentials: Path to a service account JSON file, inline JSON string,
                or a dict (e.g. from env on Streamlit Cloud).
            aws_key_id (str): AWS Access Key ID.
            aws_secret (str): AWS Secret Access Key.
            aws_region (str): Região da AWS (ex.: 'us-east-1').
        """
        # Configurações do S3
        if aws_key_id and aws_secret:
            self.s3_kwargs = {
                "key": aws_key_id,
                "secret": aws_secret,
                "client_kwargs": {"region_name": aws_region} if aws_region else {},
            }
        else:
            self.s3_kwargs = {}

        # GCS: gcsfs accepts a file path, JSON string, or credentials dict.
        if gcp_credentials is None:
            self.gcs_kwargs = {}
        elif isinstance(gcp_credentials, dict):
            self.gcs_kwargs = {"token": gcp_credentials}
        else:
            s = str(gcp_credentials).strip()
            if s.startswith("{"):
                self.gcs_kwargs = {"token": json.loads(s)}
            else:
                self.gcs_kwargs = {"token": s}

    def _get_fs(self, path: str):
        """
        Retorna o FileSystem apropriado baseado no prefixo do caminho.

        Args:
            path (str): Caminho do arquivo ou diretório.

        Returns:
            fsspec.AbstractFileSystem: Instância do sistema de arquivos.
        """
        if path.startswith("s3://"):
            return fsspec.filesystem("s3", **self.s3_kwargs)
        elif path.startswith("gs://"):
            return fsspec.filesystem("gcs", **self.gcs_kwargs)
        else:
            return fsspec.filesystem("file")

    def ls_dir(self, dir_path: str, pattern: Optional[str] = None) -> List[str]:
        """
        Lista os arquivos de um diretório.

        Args:
            dir_path (str): Caminho do diretório.
            pattern (Optional[str]): Regex para filtrar os arquivos.

        Returns:
            List[str]: Lista de arquivos encontrados.
        """
        try:
            fs = self._get_fs(dir_path)
            file_list = fs.ls(dir_path)

            if pattern:
                file_list = [
                    file
                    for file in file_list
                    if re.search(pattern, file.split("/")[-1])
                ]

            full_paths = [
                f"{dir_path.rstrip('/')}/{file.split('/')[-1]}" for file in file_list
            ]

            logger.info(f"Arquivos encontrados em {dir_path}: {full_paths}")
            return full_paths
        except Exception as e:
            logger.error(f"Erro ao listar diretório {dir_path}: {e}")
            raise

    def list_gcs_files_filtered(
        self,
        gs_prefix: str,
        *,
        min_size_bytes: Optional[int] = None,
        max_size_bytes: Optional[int] = None,
        suffixes: Optional[tuple[str, ...]] = None,
        sort_descending: bool = False,
        limit: Optional[int] = None,
    ) -> List[tuple[str, int]]:
        """
        Recursively list files under a gs:// prefix, optionally filtering by
        inclusive size bounds, suffix, sort order, and max count.
        Returns (gs:// URI, size).
        """
        if not gs_prefix.startswith("gs://"):
            raise ValueError("gs_prefix must start with gs://")

        fs = self._get_fs(gs_prefix)
        prefix = gs_prefix[len("gs://") :].rstrip("/")
        found = fs.find(prefix, detail=True)

        if isinstance(found, dict):
            items = list(found.items())
        elif isinstance(found, list):
            if not found:
                items = []
            elif isinstance(found[0], dict) and "name" in found[0]:
                items = [(d["name"], d) for d in found]
            else:
                items = [(p, fs.info(p)) for p in found]
        else:
            items = []

        results: List[tuple[str, int]] = []
        suffixes_l = tuple(s.lower() for s in suffixes) if suffixes else None

        for path, info in items:
            if not isinstance(info, dict):
                continue
            if info.get("type") == "directory":
                continue
            size = int(info.get("size") or info.get("Size") or 0)
            base = path.split("/")[-1]
            if not base or base.endswith("/"):
                continue
            if suffixes_l and not base.lower().endswith(suffixes_l):
                continue
            if min_size_bytes is not None and size < min_size_bytes:
                continue
            if max_size_bytes is not None and size > max_size_bytes:
                continue
            gs_uri = path if path.startswith("gs://") else f"gs://{path}"
            results.append((gs_uri, size))

        results.sort(key=lambda x: x[1], reverse=sort_descending)
        if limit is not None:
            results = results[:limit]
        return results

    def read(self, file_path: str) -> bytes:
        """
        Lê um arquivo binário.

        Args:
            file_path (str): Caminho do arquivo.

        Returns:
            bytes: Conteúdo do arquivo.
        """
        try:
            fs = self._get_fs(file_path)
            with fs.open(file_path, "rb") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Erro ao ler {file_path}: {e}")
            raise

    def read_text(self, file_path: str) -> str:
        """
        Lê um arquivo de texto.

        Args:
            file_path (str): Caminho do arquivo.

        Returns:
            str: Conteúdo do arquivo.
        """
        return self.read(file_path).decode("utf-8")

    def read_json(self, file_path: str) -> Union[Dict[str, Any], List[Any]]:
        """
        Lê um arquivo JSON.

        Args:
            file_path (str): Caminho do arquivo JSON.

        Returns:
            Union[Dict, List]: Conteúdo do JSON.
        """
        return json.loads(self.read(file_path))

    def save(self, content: bytes, file_path: str) -> None:
        """
        Salva conteúdo binário.

        Args:
            content (bytes): Conteúdo.
            file_path (str): Caminho de destino.
        """
        try:
            fs = self._get_fs(file_path)
            with fs.open(file_path, "wb") as f:
                f.write(content)
            logger.info(f"Arquivo salvo em {file_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar {file_path}: {e}")
            raise

    def save_text(self, content: str, file_path: str) -> None:
        """
        Salva texto.

        Args:
            content (str): Conteúdo de texto.
            file_path (str): Caminho de destino.
        """
        try:
            fs = self._get_fs(file_path)
            with fs.open(file_path, "w") as f:
                f.write(content)
            logger.info(f"Texto salvo em {file_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar {file_path}: {e}")
            raise

    def save_dict(
        self,
        data: Union[Dict[str, Any], List[Any]],
        file_path: str,
        format: str = "json",
    ) -> None:
        """
        Salva dados estruturados (JSON ou Parquet).

        Args:
            data (Union[Dict, List]): Dados.
            file_path (str): Caminho de destino.
            format (str): 'json' ou 'parquet'.
        """
        try:
            fs = self._get_fs(file_path)

            if format == "json":
                with fs.open(file_path, "w") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                logger.info(f"JSON salvo em {file_path}")
            elif format == "parquet":
                if isinstance(data, dict):
                    data = [data]
                df = pd.json_normalize(data)
                with fs.open(file_path, "wb") as f:
                    df.to_parquet(f, index=False)
                logger.info(f"Parquet salvo em {file_path}")
            else:
                raise ValueError(f"Formato inválido: {format}")
        except Exception as e:
            logger.error(f"Erro ao salvar {file_path}: {e}")
            raise

    def exists(self, path: str) -> bool:
        """
        Verifica se o arquivo existe.

        Args:
            path (str): Caminho.

        Returns:
            bool: True se existir, False caso contrário.
        """
        try:
            fs = self._get_fs(path)
            return fs.exists(path)
        except Exception as e:
            logger.error(f"Erro ao verificar {path}: {e}")
            return False

    def check_access(self, path: str) -> bool:
        """
        Verifica se há acesso (permissão) ao caminho fornecido.
        Funciona tanto para arquivos quanto para diretórios.

        Args:
            path (str): Caminho do arquivo ou diretório.

        Returns:
            bool: True se há acesso, False caso contrário.
        """
        try:
            fs = self._get_fs(path)

            if fs.isdir(path):
                fs.ls(path)
            elif fs.isfile(path):
                with fs.open(path, "rb") as f:
                    f.read(1)
            else:
                logger.warning(f"Caminho {path} não encontrado.")
                return False

            logger.info(f"Acesso confirmado para {path}")
            return True

        except Exception as e:
            logger.warning(f"Sem acesso ao path {path}: {e}")
            return False

    def make_public(self, file_path: str) -> Optional[str]:
        """
        Torna o arquivo acessível publicamente e retorna a URL pública (se aplicável).

        Args:
            file_path (str): Caminho do arquivo (s3:// ou gs://)

        Returns:
            Optional[str]: URL pública de acesso, se aplicável.
        """
        try:
            if file_path.startswith("s3://"):
        
                bucket, key = file_path[5:].split("/", 1)
                s3 = boto3.client(
                    "s3",
                    aws_access_key_id=self.s3_kwargs.get("key"),
                    aws_secret_access_key=self.s3_kwargs.get("secret"),
                    region_name=self.s3_kwargs.get("client_kwargs", {}).get(
                        "region_name"
                    ),
                )
                s3.put_object_acl(ACL="public-read", Bucket=bucket, Key=key)
                url = f"https://{bucket}.s3.amazonaws.com/{key}"
                logger.info(f"Arquivo tornado público: {url}")
                return url

            elif file_path.startswith("gs://"):
                from google.cloud import storage

                bucket_name, blob_name = file_path[6:].split("/", 1)
                client = storage.Client.from_service_account_json(
                    self.gcs_kwargs["token"]
                )
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                blob.make_public()
                logger.info(f"Arquivo tornado público: {blob.public_url}")
                return blob.public_url

            else:
                logger.warning("make_public não é suportado para arquivos locais.")
                return None

        except Exception as e:
            logger.error(f"Erro ao tornar público {file_path}: {e}")
            raise
