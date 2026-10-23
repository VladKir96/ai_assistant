"""Reject unauthorized or oversized bodies before FastAPI parses them."""

import secrets

from starlette.responses import JSONResponse


class PrivateBodyGuard:
    def __init__(self, app, token: str, limit: int, path_limits: dict[str, int] | None = None):
        self.app, self.token, self.limit = app, token, limit
        self.path_limits = path_limits or {}

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/v1/"):
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        limit = self.path_limits.get(scope["path"], self.limit)
        scheme, _, supplied = headers.get(b"authorization", b"").partition(b" ")
        if scheme.lower() != b"bearer" or not secrets.compare_digest(supplied, self.token.encode()):
            return await JSONResponse({"detail": "Invalid token"}, status_code=401)(
                scope, receive, send
            )
        try:
            declared = int(headers.get(b"content-length", b"0"))
            if declared < 0:
                raise ValueError
        except ValueError:
            return await JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)(
                scope, receive, send
            )
        if declared > limit:
            return await JSONResponse(
                {"detail": "Request body exceeds the upload limit"}, status_code=413
            )(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > limit:
                return await JSONResponse(
                    {"detail": "Request body exceeds the upload limit"}, status_code=413
                )(scope, receive, send)
            body.extend(chunk)
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay, send)
